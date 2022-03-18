{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module TD3 ( algorithm
           , Agent
           , mkAgent
           , saveAgent
           , π
           , q
           , q'
           , addNoise
           , train
           -- , play
           ) where

import Lib
import RPB
import TD3.Defaults

import Control.Monad
import GHC.Generics
import qualified Torch    as T
import qualified Torch.NN as NN

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Actor Network Specification
data ActorNetSpec = ActorNetSpec { pObsDim :: Int, pActDim :: Int }
    deriving (Show, Eq)

-- | Critic Network Specification
data CriticNetSpec = CriticNetSpec { qObsDim :: Int, qActDim :: Int }
    deriving (Show, Eq)

-- | Actor Network Architecture
data ActorNet = ActorNet { pLayer0 :: T.Linear
                         , pLayer1 :: T.Linear
                         , pLayer2 :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { qLayer0 :: T.Linear
                           , qLayer1 :: T.Linear
                           , qLayer2 :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 400) 
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 400     300)
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 300 pActDim)
                                             >>= weightInit wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec dim 400) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 400 300) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 300 1) 
                                               >>= weightInit' )
        where dim = qObsDim + qActDim

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> T.Tensor
π ActorNet{..} !o = a
  where
    a = T.tanh . T.linear pLayer2 
      . T.relu . T.linear pLayer1 
      . T.relu . T.linear pLayer0 
      $ o

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q CriticNet{..} !o !a = v
  where 
    x = T.cat (T.Dim $ -1) [o,a]
    v = T.linear qLayer2 . T.relu
      . T.linear qLayer1 . T.relu
      . T.linear qLayer0 $ x

-- | Convenience Function
q' :: CriticNet -> CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q' c1 c2 s a = v
  where
    q1 = q c1 s a
    q2 = q c2 s a
    v  = fst . T.minDim (T.Dim 1) T.KeepDim $ T.cat (T.Dim 1) [q1, q2]

------------------------------------------------------------------------------
-- TD3 Agent
------------------------------------------------------------------------------

-- | TD3 Agent
data Agent = Agent { φ       :: ActorNet  
                   , φ'      :: ActorNet  
                   , θ1      :: CriticNet  
                   , θ2      :: CriticNet  
                   , θ1'     :: CriticNet  
                   , θ2'     :: CriticNet  
                   , φOptim  :: T.Adam
                   , θ1Optim :: T.Adam
                   , θ2Optim :: T.Adam
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φOnline   <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    φTarget'  <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ1Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ1Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)

    let φTarget  = copySync φTarget'  φOnline
        θ1Target = copySync θ1Target' θ1Online
        θ2Target = copySync θ2Target' θ2Online
    

    let φOpt  = T.mkAdam 0 β1 β2 (NN.flattenParameters φOnline)
        θ1Opt = T.mkAdam 0 β1 β2 (NN.flattenParameters θ1Online)
        θ2Opt = T.mkAdam 0 β1 β2 (NN.flattenParameters θ2Online)

    pure $ Agent φOnline φTarget θ1Online θ2Online θ1Target θ2Target
                 φOpt            θ1Opt    θ2Opt                      

-- | Save an Agent to Disk
saveAgent :: Agent -> String -> IO ()
saveAgent Agent{..} path = head $ zipWith T.saveParams 
                                [q1o, q2o, q1t, q2t, ao, at] 
                                [pq1o, pq2o, pq1t, pq2t, pao, pat]
  where
    q1o  = T.toDependent <$> T.flattenParameters θ1
    q2o  = T.toDependent <$> T.flattenParameters θ2
    q1t  = T.toDependent <$> T.flattenParameters θ1'
    q2t  = T.toDependent <$> T.flattenParameters θ2'
    ao   = T.toDependent <$> T.flattenParameters φ
    at   = T.toDependent <$> T.flattenParameters φ'
    pq1o = path ++ "/q1o.pt"
    pq2o = path ++ "/q2o.pt"
    pq1t = path ++ "/q1t.pt"
    pq2t = path ++ "/q2t.pt"
    pao  = path ++ "/actor.pt"
    pat  = path ++ "/actor.pt"

---- | Load an Actor Net
--loadActor :: String -> Int -> Int -> IO ActorNet

---- | Load an Critic Net
--loadCritic :: String -> Int -> Int -> IO CriticNet

---- | Load an Agent
--loadCritic :: String -> IO Agent

addNoise :: Int -> T.Tensor -> IO T.Tensor
addNoise t action = do
    ε <- (σ' *) <$> normal' [l]
    let action' = T.clamp (- 1.0) 1.0 (action + ε)
    pure action'
  where
    l  = T.shape action !! 1
    d' = realToFrac decayPeriod
    t' = realToFrac t
    m  = min 1.0 (t' / d')
    σ' = toTensor $ σMin - (σMax - σMin) * m

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Int -> Int -> Int -> Agent -> ReplayBuffer T.Tensor -> IO Agent
updateStep _ _ 0 agent _ = pure agent
updateStep episode iteration epoch Agent{..} buffer@ReplayBuffer{..} = do
    ε <- normal μ' σ'
    let a' = π φ' s' + ε
    y <- T.detach $ r + γ * q' θ1' θ2' s' a'

    let v1  = q θ1 s a
        v2  = q θ2 s a
        jQ1 = T.mseLoss v1 y
        jQ2 = T.mseLoss v2 y

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tΘ1 Loss:\t" ++ show jQ1
        putStrLn $ "\tΘ2 Loss:\t" ++ show jQ2
    writeLoss episode iteration "Q1" (T.asValue jQ1 :: Float)
    writeLoss episode iteration "Q2" (T.asValue jQ2 :: Float)

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηθ
    (θ2Online', θ2Optim') <- T.runStep θ2 θ1Optim jQ2 ηθ

    let updateActor :: IO (ActorNet, T.Adam)
        updateActor = do
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tφ  Loss:\t" ++ show jφ
            writeLoss episode iteration "A" (T.asValue jφ :: Float)
            T.runStep φ φOptim jφ ηφ
          where
            a'' = π φ s
            v   = q θ1 a'' s
            jφ  = negate . T.mean $ v
        syncTargets :: IO (ActorNet, CriticNet, CriticNet)
        syncTargets = do
            φTarget'  <- softSync τ φ'  φ
            θ1Target' <- softSync τ θ1' θ1 
            θ2Target' <- softSync τ θ2' θ2 
            pure (φTarget', θ1Target', θ2Target')

    (φOnline', φOptim') <- if iteration `elem` [0,d .. numIterations]
                              then updateActor
                              else pure (φ, φOptim)

    (φTarget', θ1Target', θ2Target') <- if iteration `elem` [0,d .. numIterations]
                                           then syncTargets
                                           else pure (φ', θ1', θ2')

    let agent' = Agent φOnline' φTarget' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'           θ1Optim'  θ2Optim'

    updateStep episode iteration epoch' agent' buffer
  where
    μ'     = T.zerosLike rpbActions
    σ'     = T.onesLike μ' * σ
    epoch' = epoch - 1
    s      = rpbStates
    a      = rpbActions
    r      = rpbRewards
    s'     = rpbStates'

-- | Perform Policy Update Steps
updatePolicy :: Int -> Int -> Agent -> ReplayBuffer T.Tensor -> Int -> IO Agent
updatePolicy episode iteration !agent !buffer epochs = do

    memories <- bufferRandomSample batchSize buffer
    updateStep episode iteration epochs agent memories

-- | Evaluate Policy
evaluatePolicy :: Int -> Int -> Int -> Agent -> HymURL -> T.Tensor 
               -> ReplayBuffer T.Tensor -> T.Tensor 
               -> IO (ReplayBuffer T.Tensor, T.Tensor, T.Tensor)
evaluatePolicy _ _ 0 _ _ states buffer total = pure (buffer, states, total)
evaluatePolicy episode iteration step agent@Agent{..} envUrl states buffer total = do

    actions <- if p < warmupPeriode
                  then randomActionPool envUrl
                  else addNoise iteration (π φ states) >>= T.detach
    
    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions

    let keys   = head infos
        total' = T.cat (T.Dim 0) [total, rewards]
    
    !states' <- if T.any dones 
                then flip processGace keys <$> resetPool' envUrl dones
                else pure $ processGace states'' keys

    let !buffer' = bufferPush bufferSize buffer states actions rewards states' dones

    writeReward' episode iteration rewards

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero $ dones
        putStrLn $ "Environments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t" ++ show de

    evaluatePolicy episode iteration step' agent envUrl states' buffer' total'
  where
    p       = iteration * numEnvs
    step'   = step - 1

-- | Run Twin Delayed Deep Deterministic Policy Gradient Training
runAlgorithm :: Int -> Int -> Agent -> HymURL -> Bool -> ReplayBuffer T.Tensor
             -> T.Tensor -> T.Tensor -> IO Agent
runAlgorithm episode iteration agent _ True _ _ rewards = do
    putStrLn $ "Episode " ++ show episode ++ " done after " ++ show iteration 
            ++ " iterations, with a total reward of " ++ show reward'
    pure agent
  where
    --reward' = T.asValue . T.sumAll $ rewards :: Float
    reward' = T.asValue rewards :: Float

runAlgorithm episode iteration agent envUrl _ buffer states total = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Episode " ++ show episode ++ ", Iteration " ++ show iteration

    (!memories', !states', !rewards) <- evaluatePolicy episode iteration numSteps 
                                                   agent envUrl states buffer total

    let !reward' = total + T.sumAll rewards
        !buffer' = bufferPush' bufferSize buffer memories'

    !agent' <- if bufferLength buffer' < batchSize 
                  then pure agent
                  else updatePolicy episode iteration agent buffer' numEpochs

    runAlgorithm episode iteration' agent' envUrl done' buffer' states' reward'
  where
    done'      = iteration >= numIterations
    iteration' = iteration + 1

-- | Train Twin Delayed Deep Deterministic Policy Gradient Agent on Environment
train :: Int -> Int -> HymURL -> IO Agent
train obsDim actDim envUrl = do
    remoteLogPath envUrl >>= setupLogging 

    !agent <- mkAgent obsDim actDim >>= foldLoop' numEpisodes
        (\agent' episode -> do
            states' <- toFloatGPU <$> resetPool envUrl
            keys <- infoPool envUrl

            let !states    = processGace states' keys
                !buffer = makeBuffer
                !rewards = emptyTensor

            runAlgorithm episode 0 agent' envUrl False buffer states rewards)
    saveAgent agent ptPath
    pure agent
  where 
      ptPath = "./models/" ++ algorithm

-- | Play Environment with Twin Delayed Deep Deterministic Policy Gradient Agent
--play :: Agent -> HymURL -> IO (M.Map String Float, Float)
