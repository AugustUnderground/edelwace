{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Soft Actor Critic Algorithm Defaults
module SAC ( algorithm
           , Agent (..)
           , mkAgent
           , saveAgent
           , loadAgent
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
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 256) 
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256     256)
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
                                             >>= weightInit wInit )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
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
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
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

    pure $ Agent φOnline θ1Online θ2Online θ1Target θ2Target 
                 φOpt    θ1Opt    θ2Opt  
                 hTarget αlog     αOpt
  where
    hTarget = realToFrac . negate $ actDim

-- | Save an Agent Checkpoint
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = do

        T.saveParams φ   (path ++ "/actor.pt")
        T.saveParams θ1  (path ++ "/q1Online.pt")
        T.saveParams θ2  (path ++ "/q2Online.pt")
        T.saveParams θ1' (path ++ "/q1Target.pt")
        T.saveParams θ2' (path ++ "/q2Target.pt")
        T.save       [α] (path ++ "/alpha.pt")

        saveOptim φOptim  (path ++ "/actorOptim")
        saveOptim θ1Optim (path ++ "/q1Optim")
        saveOptim θ2Optim (path ++ "/q2Optim")
        saveOptim αOptim  (path ++ "/alphaOptim")

        putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "
  where
    α = T.toDependent αLog

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim iter actDim = do
        Agent{..} <- mkAgent obsDim actDim

        fφ    <- T.loadParams φ   (path ++ "/actor.pt")
        fθ1   <- T.loadParams θ1  (path ++ "/q1Online.pt")
        fθ2   <- T.loadParams θ2  (path ++ "/q2Online.pt")
        fθ1'  <- T.loadParams θ1' (path ++ "/q1Target.pt")
        fθ2'  <- T.loadParams θ2' (path ++ "/q2Target.pt")
        fαLog <- T.load (path ++ "/alpha.pt") >>= T.makeIndependent . head

        fφOpt  <- loadOptim iter β1 β2 (path ++ "/actorOptim")
        fθ1Opt <- loadOptim iter β1 β2 (path ++ "/q1Optim")
        fθ2Opt <- loadOptim iter β1 β2 (path ++ "/q2Optim")
        fαOpt  <- loadOptim iter β1 β2 (path ++ "/alphaOptim")
       
        pure $ Agent fφ fθ1 fθ2 fθ1' fθ2' fφOpt fθ1Opt fθ2Opt h' fαLog fαOpt

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
updateStep :: Int -> Int -> Agent -> ReplayBuffer T.Tensor -> T.Tensor 
           -> T.Tensor -> IO (Agent, T.Tensor)
updateStep _ 0 agent _ _ prios = pure (agent, prios)
updateStep iteration epoch agent@Agent{..} memories@ReplayBuffer{..} weights _ = do
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

    writeLoss iteration "Q1" (T.asValue jQ1 :: Float)
    writeLoss iteration "Q2" (T.asValue jQ2 :: Float)
        
    (a_t0', logπ_t0') <- evaluate agent s_t0 εNoise

    let updateAlpha :: IO(T.IndependentTensor, T.Adam)
        updateAlpha = do
            logπ_t0 <- T.clone logπ_t0' >>= T.detach
            let jα = T.mean (- α * logπ_t0 - α * h)
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tα  Loss:\t" ++ show jα
            writeLoss iteration "A" (T.asValue jα :: Float)
            T.runStep αLog αOptim jα ηα
        updateActor :: IO(ActorNet, T.Adam)
        updateActor = do
            q_t0' <- T.detach $ q' θ1 θ2 s_t0 a_t0'
            let jπ = T.mean ((α' * logπ_t0') - q_t0')
            when (verbose && iteration `elem` [0,10 .. numIterations]) do
                putStrLn $ "\tπ  Loss:\t" ++ show jπ
            writeLoss iteration "P" (T.asValue jπ :: Float)
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

    updateStep iteration epoch' agent' memories weights prios'
  where
    epoch' = epoch - 1
    s_t0   = rpbStates
    a_t0   = rpbActions
    s_t1   = rpbStates'
    d'     = toFloatGPU 1.0 - rpbDones
    w      = T.reshape [-1,1] weights
    r      = rpbRewards * toTensor rewardScale
    h      = toTensor h'
    --r    = scaleRewards rewards ρ

-- | Perform Policy Update Steps
updatePolicy :: Int -> Agent -> PERBuffer T.Tensor -> Int 
             -> IO (PERBuffer T.Tensor, Agent)
updatePolicy iteration !agent !buffer epochs = do
    (memories, indices, weights) <- perSample buffer iteration batchSize
    let prios = perPriorities buffer
    (agent', prios') <- updateStep iteration epochs agent 
                                   memories weights prios
    let buffer' = perUpdate buffer indices prios'
    pure (buffer', agent')

-- | Take steps in the Environment, evaluating the current policy
evaluatePolicy :: Int -> Int -> Agent -> HymURL -> PERBuffer T.Tensor
               -> T.Tensor -> IO (PERBuffer T.Tensor, T.Tensor)
evaluatePolicy _ 0 _ _ buffer states = pure (buffer, states)
evaluatePolicy iteration step agent envUrl buffer states = do

    actions <- act agent states
    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions

    writeReward iteration rewards
 
    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "\tEnvironments finished episode after " ++ show iteration 
                ++ " iterations, resetting:\n\t\t" ++ show de
   
    let keys = head infos
    !states' <- if T.any dones 
                   then flip processGace keys <$> resetPool' envUrl dones
                   else pure $ processGace states'' keys

    let buffer' = perPush buffer states actions rewards states' dones

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    evaluatePolicy iteration step' agent envUrl buffer' states'
  where
    step' = step - 1

-- | Run Soft Actor Critic Training
runAlgorithm :: Int -> Agent -> HymURL -> Bool -> PERBuffer T.Tensor
             -> T.Tensor -> IO Agent
runAlgorithm _ agent _ True _ _ = pure agent
runAlgorithm iteration agent envUrl _ buffer states = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!memories', !states') <- evaluatePolicy iteration numSteps agent 
                                             envUrl buffer states

    let !buffer'' = perPush' buffer memories'
        !bufLen   = bufferLength . perMemories $ buffer''

    (!buffer', !agent') <- if bufLen < batchSize 
                              then pure (buffer'', agent)
                              else updatePolicy iteration agent buffer'' numEpochs
    
    when (iteration `elem` [0,10 .. numIterations]) do
        saveAgent ptPath agent 

    runAlgorithm iteration' agent' envUrl done' buffer' states'
  where
    done'      = iteration >= numIterations
    iteration' = iteration + 1
    ptPath     = "./models/" ++ algorithm

-- | Train Soft Actor Critic Agent on Environment
train :: Int -> Int -> HymURL -> IO Agent
train obsDim actDim envUrl = do
    remoteLogPath envUrl >>= setupLogging 

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states    = processGace states' keys
        !buffer = makePERBuffer bufferSize αStart βStart βFrames

    !agent <- mkAgent obsDim actDim >>= (\agent' -> 
        runAlgorithm 0 agent' envUrl False buffer states)

    saveAgent ptPath agent 
    pure agent
  where 
    ptPath = "./models/" ++ algorithm

-- | Play Environment with Soft Actor Critic Agent
-- play :: Agent -> HymURL -> IO Agent
