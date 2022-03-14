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
           , actorForward
           , criticForward
           , criticForward'
           , act
           , evaluate
           , train
           -- , play
           ) where

import Lib
import PER
import SAC.Defaults

import Control.Monad
import GHC.Generics
import qualified Data.Random as RNG
import qualified Torch              as T
import qualified Torch.Initializers as T (xavierNormal)
import qualified Torch.NN           as NN

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Calculate weight Limits based on Layer Dimensions
weightLimit :: T.Linear -> Float
weightLimit layer = fanIn ** (- 0.5)
  where
    fanIn = realToFrac . head . T.shape . T.toDependent . NN.weight $ layer

-- | Initialize Weights of Linear Layer
weightInit :: Float -> T.Linear -> IO T.Linear
weightInit limit layer = do
    weight' <- T.xavierNormal limit dims >>= T.makeIndependent
    pure T.Linear { NN.weight = weight', NN.bias = bias' }
  where
    dims  = T.shape . T.toDependent . NN.weight $ layer
    bias' = NN.bias layer

-- | Initialize weights based on Fan In
weightInit' :: T.Linear -> IO T.Linear
weightInit' layer = weightInit limit layer
  where
    limit = weightLimit layer

-- | Actor Network Specification
data ActorNetSpec = ActorNetSpec { pObsDim :: Int, pActDim :: Int }
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
    sample ActorNetSpec {..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 256) 
                                              >>= weightInit' )
                                        <*> ( T.sample (T.LinearSpec 256     256)
                                              >>= weightInit' )
                                        <*> ( T.sample (T.LinearSpec 256 pActDim)
                                              >>= weightInit wInit )
                                        <*> ( T.sample (T.LinearSpec 256 pActDim)
                                              >>= weightInit wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec {..} = CriticNet <$> ( T.sample (T.LinearSpec dim 256) 
                                                >>= weightInit' )
                                          <*> ( T.sample (T.LinearSpec 256 256) 
                                                >>= weightInit' )
                                          <*> ( T.sample (T.LinearSpec 256 1) 
                                                >>= weightInit' )
        where dim = qObsDim + qActDim

-- | Actor Network Forward Pass
actorForward :: ActorNet -> T.Tensor -> (T.Tensor, T.Tensor)
actorForward ActorNet {..} !obs = (μ, σ)
  where
    !x = T.linear pLayer1 . T.relu
       . T.linear pLayer0 $ obs
    !μ = T.linear pLayerμ x
    !σ = T.clamp σMin σMax . T.linear pLayerσ $ x

-- | Critic Network Forward Pass
criticForward :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
criticForward CriticNet {..} !o !a = T.linear qLayer2 . T.relu
                                   . T.linear qLayer1 . T.relu
                                   . T.linear qLayer0 $ input
  where
    !input = T.cat (T.Dim $ -1) [o,a]

-- | Convenience Function
criticForward' :: CriticNet -> CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
criticForward' !c1 !c2 !s !a = q
  where
    !q1 = criticForward c1 s a
    !q2 = criticForward c2 s a
    !q  = fst . T.minDim (T.Dim 1) T.KeepDim $ T.cat (T.Dim 1) [q1, q2]

------------------------------------------------------------------------------
-- SAC Agent
------------------------------------------------------------------------------

-- | SAC Agent
data Agent = Agent { actorOnline   :: ActorNet
                   , critic1Online :: CriticNet  
                   , critic2Online :: CriticNet  
                   , critic1Target :: CriticNet  
                   , critic2Target :: CriticNet  
                   , actorOptim    :: T.Adam
                   , critic1Optim  :: T.Adam
                   , critic2Optim  :: T.Adam
                   , entropyTarget :: Float
                   , alphaLog      :: T.IndependentTensor
                   , alphaOptim    :: T.Adam
                   } deriving (Generic, Show)

-- | Agent constructor
makeAgent :: Int -> Int -> IO Agent
makeAgent obsDim actDim = do
    πOnline   <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    q1Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    q2Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    q1Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    q2Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)

    αlog <- T.makeIndependent . toFloatGPU $ T.zeros' [1]

    let q1Target = copySync q1Target' q1Online
        q2Target = copySync q2Target' q2Online

    let πOptim  = T.mkAdam 0 β1 β2 (NN.flattenParameters πOnline)
        q1Optim = T.mkAdam 0 β1 β2 (NN.flattenParameters q1Online)
        q2Optim = T.mkAdam 0 β1 β2 (NN.flattenParameters q2Online)
        αOptim  = T.mkAdam 0 β1 β2 (NN.flattenParameters [αlog])

    let entropyTarget' = realToFrac (- actDim)

    pure $ Agent πOnline q1Online q2Online q1Target q2Target 
                 πOptim  q1Optim  q2Optim  
                 entropyTarget' αlog αOptim

-- | Get an Action (no grad)
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent {..} !state = do
    ε' <- toTensor <$> RNG.sample n
    T.detach . T.tanh $ (μ + σ * ε')
  where
    (μ,σ') = actorForward actorOnline state
    σ      = T.exp σ'
    n      = RNG.Normal 0 1 :: RNG.Normal Float

-- | Get an action and log probs (grad)
evaluate :: Agent -> T.Tensor -> Float -> IO (T.Tensor, T.Tensor)
evaluate Agent {..} !state εN = do
    ε' <- toTensor <$> RNG.sample n
    z' <- mapM (mapM RNG.sample) n''
    let a  = T.tanh (μ + σ * ε')
        l1 = toTensor (zipWith (zipWith RNG.logPdf) n'' z')
        l2 = T.log $ 1.0 - T.pow (2.0 :: Float) a + T.asTensor εN
        p  = l1 - l2
    pure (a,p)
  where
    (μ,σ') = actorForward actorOnline state
    σ      = T.exp σ'
    n      = RNG.Normal 0 1 :: RNG.Normal Float
    σ''    = T.asValue σ :: [[Float]]
    μ''    = T.asValue μ :: [[Float]]
    n''    = zipWith (zipWith RNG.Normal) μ'' σ'' :: [[RNG.Normal Float]]

-- | Save an Agent to Disk
saveAgent :: Agent -> String -> IO ()
saveAgent Agent {..} path = head $ zipWith T.saveParams 
                                [ao, q1o, q2o, q1t, q2t] 
                                [pao, pq1o, pq2o, pq1t, pq2t]
  where
    ao   = T.toDependent <$> T.flattenParameters actorOnline
    q1o  = T.toDependent <$> T.flattenParameters critic1Online
    q2o  = T.toDependent <$> T.flattenParameters critic2Online
    q1t  = T.toDependent <$> T.flattenParameters critic1Target
    q2t  = T.toDependent <$> T.flattenParameters critic2Target
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

-- | Softly update parameters from Online Net to Target Net
softUpdate :: T.Tensor -> T.Tensor -> T.Tensor
softUpdate t o = (t * (o' - τ')) + (o * τ')
  where
    τ' = toTensor τSoft
    o' = T.onesLike τ'

-- | Softly copy parameters from Online Net to Target Net
softSync :: CriticNet -> CriticNet -> IO CriticNet
softSync target online =  NN.replaceParameters target 
                        <$> mapM T.makeIndependent tUpdate 
  where
    tParams = fmap T.toDependent . NN.flattenParameters $ target
    oParams = fmap T.toDependent . NN.flattenParameters $ online
    tUpdate = zipWith softUpdate tParams oParams

-- | Hard Copy of Parameter from one net to the other
copySync :: CriticNet -> CriticNet -> CriticNet
copySync target =  NN.replaceParameters target . NN.flattenParameters

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Int -> Int -> Int -> Agent -> ReplayBuffer -> T.Tensor -> T.Tensor 
           -> IO (Agent, T.Tensor)
updateStep _ _ 0 agent _ _ prios = pure (agent, prios)
updateStep episode iteration epoch agent@Agent {..} memories@ReplayBuffer {..} weights _ = do
    (a_t1, logπ_t1) <- evaluate agent s_t1 εNoise
    let !logπ_t1' = T.meanDim (T.Dim 1) T.KeepDim T.Float logπ_t1
        !α        = T.exp . T.toDependent $ alphaLog
    α' <- if iteration == 0 then pure (toFloatGPU 0.0)
                            else T.detach α

    let q_t1' = criticForward' critic1Target critic2Target s_t1 a_t1
    q_t1 <- T.detach (r + (γ' * d' * (q_t1' - α' * logπ_t1'))) >>= T.clone

    let q1_t0 = criticForward critic1Online s_t0 a_t0
        q2_t0 = criticForward critic2Online s_t0 a_t0

    let δ1    = w * T.pow (2.0 :: Float) (q1_t0 - q_t1)
        δ2    = w * T.pow (2.0 :: Float) (q2_t0 - q_t1)

    jQ1 <- T.clone $ T.mean (0.5 * δ1)
    jQ2 <- T.clone $ T.mean (0.5 * δ2)

    (critic1Online', critic1Optim') <- T.runStep critic1Online critic1Optim jQ1 ηq
    (critic2Online', critic2Optim') <- T.runStep critic2Online critic2Optim jQ2 ηq
    
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
            T.runStep alphaLog alphaOptim jα ηα
        updateActor :: IO(ActorNet, T.Adam)
        updateActor = do
            q_t0' <- T.detach $ criticForward' critic1Online critic2Online s_t0 a_t0'
            let jπ = T.mean ((α' * logπ_t0') - q_t0')
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tπ  Loss:\t" ++ show jπ
            writeLoss episode iteration "P" (T.asValue jπ :: Float)
            T.runStep actorOnline actorOptim jπ ηπ
        syncCritic :: IO (CriticNet, CriticNet)
        syncCritic = do
            critic1Target' <- softSync critic1Target critic1Online 
            critic2Target' <- softSync critic2Target critic2Online 
            pure (critic1Target', critic2Target')

    (alphaLog', alphaOptim') <- if iteration `elem` [0,d .. numIterations]
                                   then updateAlpha
                                   else pure (alphaLog, alphaOptim)

    (actorOnline', actorOptim') <- if iteration `elem` [0,d .. numIterations]
                                      then updateActor
                                      else pure (actorOnline, actorOptim)

    (critic1Target', critic2Target') <- if iteration `elem` [0,d .. numIterations]
                                           then syncCritic
                                           else pure (critic1Target, critic2Target)

    prios' <- T.detach $ T.abs (0.5 * (δ1 + δ2) + ε')

    let agent' = Agent actorOnline' critic1Online' critic2Online'
                       critic1Target' critic2Target' actorOptim' critic1Optim' 
                       critic2Optim' entropyTarget alphaLog' alphaOptim'

    updateStep episode iteration epoch' agent' memories weights prios'
  where
    epoch' = epoch - 1
    s_t0 = states
    a_t0 = actions
    s_t1 = states'
    d'   = toFloatGPU 1.0 - dones
    w    = T.reshape [-1,1] weights
    r    = rewards * toTensor rewardScale
    γ'   = toTensor γ
    h    = toTensor entropyTarget
    ε'   = toTensor εConst
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
evaluateStep :: Int -> Int -> Int -> Agent -> HymURL -> PERBuffer -> T.Tensor 
             -> T.Tensor -> IO (PERBuffer, T.Tensor, T.Tensor)
evaluateStep _ _ 0 _ _ buffer obs total = pure (buffer, obs, total)
evaluateStep episode iteration step agent envUrl buffer obs total = do
    actions <- act agent obs
    (obs_, rewards, dones, infos) <- stepPool envUrl actions

    let keys    = head infos
        obs''   = processGace obs_ keys
        buffer' = perPush buffer obs actions rewards obs'' dones
        total'  = T.cat (T.Dim 0) [total, rewards]
    
    obs' <- if T.any dones 
               then flip processGace keys <$> resetPool' envUrl dones
               else pure obs''

    writeReward' episode iteration rewards

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero $ dones
        putStrLn $ "Environments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t" ++ show de

    evaluateStep episode iteration step' agent envUrl buffer' obs' total'
  where
    step' = step - 1

-- | Evaluate the current policy in the given Environment
evaluatePolicy :: Int -> Int -> Agent -> HymURL -> PERBuffer -> T.Tensor -> Int
               -> IO (PERBuffer, T.Tensor, T.Tensor)
evaluatePolicy episode iteration !agent envUrl !buffer obs steps = do
    evaluateStep episode iteration steps agent envUrl buffer obs total
  where
    total = emptyTensor
    
-- | Run Soft Actor Critic Training
runAlgorithm :: Int -> Int -> Agent -> HymURL -> Bool -> PERBuffer -> T.Tensor 
             -> T.Tensor -> IO Agent
runAlgorithm episode iteration agent _ True _ _ reward = do
    putStrLn $ "Episode " ++ show episode ++ " done after " ++ show iteration 
            ++ " iterations, with a total reward of " ++ show reward'
    saveAgent agent ptPath
    pure agent
  where
    reward' = T.asValue . T.sumAll $ reward :: Float
    ptPath  = "./models/" ++ algorithm

runAlgorithm episode iteration !agent envUrl _ !buffer obs total = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Episode " ++ show episode ++ ", Iteration " ++ show iteration
    
    (!memories', !obs', !reward) <- evaluatePolicy episode iteration agent 
                                                   envUrl buffer obs numSteps

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

-- | Train Soft Actor Critic Agent
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
