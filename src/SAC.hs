{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module SAC ( algorithm
           , Agent
           , makeAgent
           , saveAgent
           , π
           , q
           , q'
           , act
           , evaluate
           , train
           -- , play
           ) where

import Lib
import RPB
import SAC.Defaults

import Control.Monad
import GHC.Generics
import qualified Torch    as T
import qualified Torch.NN as NN

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Actor Network Specification
data ActorNetSpec = ActorNetSpec { aObsDim :: Int, aActDim :: Int }
    deriving (Show, Eq)

-- | Critic Network Specification
data CriticNetSpec = CriticNetSpec { qObsDim :: Int, qActDim :: Int }
    deriving (Show, Eq)

-- | Actor Network Architecture
data ActorNet = ActorNet { pLayer0 :: T.Linear
                         , pLayer1 :: T.Linear
                         , pLayerμ :: T.Linear
                         , pLayerσ :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { qLayer0 :: T.Linear
                           , qLayer1 :: T.Linear
                           , qLayer2 :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec aObsDim 256) 
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256     256)
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256 aActDim)
                                             >>= weightInit wInit )
                                       <*> ( T.sample (T.LinearSpec 256 aActDim)
                                             >>= weightInit wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec dim 256) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 256 256) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 256 1) 
                                               >>= weightInit' )
        where dim = qObsDim + qActDim

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> (T.Tensor, T.Tensor)
π ActorNet{..} s = (μ, σ)
  where
    x = T.linear pLayer1 . T.relu
      . T.linear pLayer0 $ s
    μ = T.linear pLayerμ x
    σ = T.clamp σMin σMax . T.linear pLayerσ $ x

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q CriticNet{..} s a = T.linear qLayer2 . T.relu
                    . T.linear qLayer1 . T.relu
                    . T.linear qLayer0 $ x
  where
    x = T.cat (T.Dim $ -1) [s,a]

-- | Convenience Function
q' :: CriticNet -> CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q' c1 c2 s a = fst . T.minDim (T.Dim 1) T.KeepDim $ T.cat (T.Dim 1) [q1, q2]
  where
    q1 = q c1 s a
    q2 = q c2 s a

------------------------------------------------------------------------------
-- SAC Agent
------------------------------------------------------------------------------

-- | SAC Agent
data Agent = Agent { φ       :: ActorNet
                   , θ1      :: CriticNet  
                   , θ2      :: CriticNet  
                   , θ1'     :: CriticNet  
                   , θ2'     :: CriticNet  
                   , φOptim  :: T.Adam
                   , θ1Optim :: T.Adam
                   , θ2Optim :: T.Adam
                   , h'      :: Float
                   , αLog    :: T.IndependentTensor
                   , αOptim  :: T.Adam
                   } deriving (Generic, Show)

-- | Agent constructor
makeAgent :: Int -> Int -> IO Agent
makeAgent obsDim actDim = do
    φOnline   <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ1Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ1Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)

    αlog <- T.makeIndependent . toFloatGPU $ T.zeros' [1]

    let θ1Target = copySync θ1Target' θ1Online
        θ2Target = copySync θ2Target' θ2Online

    let φOpt  = T.mkAdam 0 β1 β2 (NN.flattenParameters φOnline)
        θ1Opt = T.mkAdam 0 β1 β2 (NN.flattenParameters θ1Online)
        θ2Opt = T.mkAdam 0 β1 β2 (NN.flattenParameters θ2Online)
        αOpt  = T.mkAdam 0 β1 β2 (NN.flattenParameters [αlog])

    let hTarget = realToFrac (- actDim)

    pure $ Agent φOnline θ1Online θ2Online θ1Target θ2Target 
                 φOpt    θ1Opt    θ2Opt  
                 hTarget αlog     αOpt

-- | Save an Agent to Disk
saveAgent :: Agent -> String -> IO ()
saveAgent Agent{..} path = head $ zipWith T.saveParams 
                                [ao, q1o, q2o, q1t, q2t] 
                                [pao, pq1o, pq2o, pq1t, pq2t]
  where
    ao   = T.toDependent <$> T.flattenParameters φ
    q1o  = T.toDependent <$> T.flattenParameters θ1
    q2o  = T.toDependent <$> T.flattenParameters θ2
    q1t  = T.toDependent <$> T.flattenParameters θ1'
    q2t  = T.toDependent <$> T.flattenParameters θ2'
    pao  = path ++ "/actor.pt"
    pq1o = path ++ "/q1o.pt"
    pq2o = path ++ "/q2o.pt"
    pq1t = path ++ "/q1t.pt"
    pq2t = path ++ "/q2t.pt"

---- | Load an Actor Net
--loadActor :: String -> Int -> Int -> IO ActorNet
--loadActor fp numObs numAct = T.sample (ActorNetSpec numObs numAct) 
--                           >>= flip T.loadParams fp

---- | Load an Critic Net
--loadCritic :: String -> Int -> Int -> IO CriticNet
--loadCritic fp numObs numAct = T.sample (CriticNetSpec numObs numAct) 
--                            >>= flip T.loadParams fp

---- | Load an Agent
--loadCritic :: String -> IO Agent

-- | Get an Action (no grad)
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} !s = do
    ε <- normal' [1]
    T.detach . T.tanh $ (μ + σ * ε)
  where
    (μ,σ') = π φ s
    σ      = T.exp σ'

-- | Get an action and log probs (grad)
evaluate :: Agent -> T.Tensor -> T.Tensor -> IO (T.Tensor, T.Tensor)
evaluate Agent{..} !s εN = do
    ε <- normal' [1]
    z' <- sample n
    let a  = T.tanh (μ + σ * ε)
        l1 = logProb n z'
        l2 = T.log $ 1.0 - T.pow (2.0 :: Float) a + εN
        p  = l1 - l2
    pure (a,p)
  where
    (μ,σ') = π φ s
    σ      = T.exp σ'
    n      = Normal μ σ

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Int -> Int -> Int -> Agent -> ReplayBuffer -> T.Tensor -> T.Tensor 
           -> IO (Agent, T.Tensor)
updateStep _ _ 0 agent _ _ prios = pure (agent, prios)
updateStep episode iteration epoch agent@Agent{..} memories@ReplayBuffer{..} weights _ = do
    (a_t1, logπ_t1) <- evaluate agent s_t1 εNoise
    let !logπ_t1' = T.meanDim (T.Dim 1) T.KeepDim T.Float logπ_t1
        !α        = T.exp . T.toDependent $ αLog
    α' <- if iteration == 0 then pure (toFloatGPU 0.0)
                            else T.detach α

    let q_t1' = q' θ1' θ2' s_t1 a_t1
    q_t1 <- T.detach (r + (γ * d' * (q_t1' - α' * logπ_t1'))) >>= T.clone

    let q1_t0 = q θ1 s_t0 a_t0
        q2_t0 = q θ2 s_t0 a_t0

    let δ1    = w * T.pow (2.0 :: Float) (q1_t0 - q_t1)
        δ2    = w * T.pow (2.0 :: Float) (q2_t0 - q_t1)

    jQ1 <- T.clone $ T.mean (0.5 * δ1)
    jQ2 <- T.clone $ T.mean (0.5 * δ2)

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηq
    (θ2Online', θ2Optim') <- T.runStep θ2 θ2Optim jQ2 ηq
    
    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tQ1 Loss:\t" ++ show jQ1
        putStrLn $ "\tQ2 Loss:\t" ++ show jQ2

    writeLoss episode iteration "Q1" (T.asValue jQ1 :: Float)
    writeLoss episode iteration "Q2" (T.asValue jQ2 :: Float)
        
    (a_t0', logπ_t0') <- evaluate agent s_t0 εNoise

    let updateAlpha :: IO(T.IndependentTensor, T.Adam)
        updateAlpha = do
            logπ_t0 <- T.clone logπ_t0' >>= T.detach
            let jα = T.mean (- α * logπ_t0 - α * h)
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tα  Loss:\t" ++ show jα
            writeLoss episode iteration "A" (T.asValue jα :: Float)
            T.runStep αLog αOptim jα ηα
        updateActor :: IO(ActorNet, T.Adam)
        updateActor = do
            q_t0' <- T.detach $ q' θ1 θ2 s_t0 a_t0'
            let jπ = T.mean ((α' * logπ_t0') - q_t0')
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tπ  Loss:\t" ++ show jπ
            writeLoss episode iteration "P" (T.asValue jπ :: Float)
            T.runStep φ φOptim jπ ηπ
        syncCritic :: IO (CriticNet, CriticNet)
        syncCritic = do
            θ1Target' <- softSync τ θ1' θ1 
            θ2Target' <- softSync τ θ2' θ2 
            pure (θ1Target', θ2Target')

    (αlog', αOptim') <- if iteration `elem` [0,d .. numIterations]
                                   then updateAlpha
                                   else pure (αLog, αOptim)

    (φOnline', φOptim') <- if iteration `elem` [0,d .. numIterations]
                                      then updateActor
                                      else pure (φ, φOptim)

    (θ1Target', θ2Target') <- if iteration `elem` [0,d .. numIterations]
                                           then syncCritic
                                           else pure (θ1', θ2')

    prios' <- T.detach $ T.abs (0.5 * (δ1 + δ2) + εConst)

    let agent' = Agent φOnline' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'  θ1Optim'  θ2Optim'  h' αlog' αOptim'

    updateStep episode iteration epoch' agent' memories weights prios'
  where
    epoch' = epoch - 1
    s_t0   = states
    a_t0   = actions
    s_t1   = states'
    d'     = toFloatGPU 1.0 - dones
    w      = T.reshape [-1,1] weights
    r      = rewards * toTensor rewardScale
    h      = toTensor h'
    --r    = scaleRewards rewards ρ

-- | Perform Policy Update Steps
updatePolicy :: Int -> Int -> Agent -> PERBuffer -> Int -> IO (PERBuffer, Agent)
updatePolicy episode iteration !agent !buffer epochs = do
    (memories, indices, weights) <- perSample buffer iteration batchSize
    let prios = priorities buffer
    (agent', prios') <- updateStep episode iteration epochs agent 
                                   memories weights prios
    let buffer' = perUpdate buffer indices prios'
    pure (buffer', agent')

-- | Take steps in the Environment, evaluating the current policy
evaluatePolicy :: Int -> Int -> Int -> Agent -> HymURL -> PERBuffer -> T.Tensor 
               -> T.Tensor -> IO (PERBuffer, T.Tensor, T.Tensor)
evaluatePolicy _ _ 0 _ _ buffer obs total = pure (buffer, obs, total)
evaluatePolicy episode iteration step agent envUrl buffer obs total = do

    actions <- act agent obs
    (!obs'', !rewards, !dones, !infos) <- stepPool envUrl actions

    let keys    = head infos
        total'  = T.cat (T.Dim 0) [total, rewards]
 
    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "Environments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t" ++ show de
   
    !obs' <- if T.any dones 
                then flip processGace keys <$> resetPool' envUrl dones
                else pure $ processGace obs'' keys

    let buffer' = perPush buffer obs actions rewards obs' dones

    writeReward' episode iteration rewards

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    evaluatePolicy episode iteration step' agent envUrl buffer' obs' total'
  where
    step' = step - 1

-- | Run Soft Actor Critic Training
runAlgorithm :: Int -> Int -> Agent -> HymURL -> Bool -> PERBuffer -> T.Tensor 
             -> T.Tensor -> IO Agent
runAlgorithm episode iteration agent _ True _ _ reward = do
    putStrLn $ "Episode " ++ show episode ++ " done after " ++ show iteration 
            ++ " iterations, with a total reward of " ++ show reward'
    pure agent
  where
    reward' = T.asValue . T.sumAll $ reward :: Float

runAlgorithm episode iteration agent envUrl _ buffer obs total = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Episode " ++ show episode ++ ", Iteration " ++ show iteration
    
    let total' = emptyTensor
    (!memories', !obs', !reward) <- evaluatePolicy episode iteration numSteps
                                                   agent envUrl buffer obs total'

    let !reward'  = T.cat (T.Dim 1) [total, reward]
        !buffer'' = perPush' buffer memories'
        !bufLen   = bufferLength . memories $ buffer''

    (!buffer', !agent') <- if bufLen < batchSize 
                              then pure (buffer'', agent)
                              else updatePolicy episode iteration agent 
                                                buffer'' numEpochs

    runAlgorithm episode iteration' agent' envUrl done' buffer' obs' reward'
  where
    done'      = iteration >= numIterations
    iteration' = iteration + 1

-- | Train Soft Actor Critic Agent on Environment
train :: Int -> Int -> HymURL -> IO Agent
train obsDim actDim envUrl = do
    remoteLogPath envUrl >>= setupLogging 

    !agent <- makeAgent obsDim actDim >>= foldLoop' numEpisodes
        (\agent' episode -> do
            obs' <- toFloatGPU <$> resetPool envUrl
            keys <- infoPool envUrl

            let !obs    = processGace obs' keys
                !buffer = makePERBuffer bufferSize αStart βStart βFrames
                !reward = emptyTensor

            runAlgorithm episode 0 agent' envUrl False buffer obs reward)
    saveAgent agent ptPath
    pure agent
  where 
      ptPath = "./models/" ++ algorithm

-- | Play Environment with Soft Actor Critic Agent
-- play :: Agent -> HymURL -> IO Agent
