{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Twin Delayed Deep Deterministic Policy Gradient Algorithm
module TD3 ( algorithm
           , ActorNetSpec (..)
           , CriticNetSpec (..)
           , ActorNet (..)
           , CriticNet (..)
           , Agent (..)
           , mkAgent
           , saveAgent
           , loadAgent
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
import MLFlow       (TrackingURI)

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
                         , pLayer2 :: T.Linear 
                         } deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { qLayer0 :: T.Linear
                           , qLayer1 :: T.Linear
                           , qLayer2 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 400) 
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 400     300)
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 300 pActDim)
                                             >>= weightInitUniform (-wInit) wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec dim 400) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 400 300) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 300 1) 
                                               >>= weightInitUniform' )
        where dim = qObsDim + qActDim

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> T.Tensor
π ActorNet{..} o = a
  where
    a = T.tanh . T.linear pLayer2 
      . T.relu . T.linear pLayer1 
      . T.relu . T.linear pLayer0 
      $ o

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q CriticNet{..} o a = v
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
data Agent = Agent { φ       :: ActorNet    -- ^ Online Policy φ
                   , φ'      :: ActorNet    -- ^ Target Policy φ'
                   , θ1      :: CriticNet   -- ^ Online Critic θ1
                   , θ2      :: CriticNet   -- ^ Online Critic θ2
                   , θ1'     :: CriticNet   -- ^ Target Critic θ1
                   , θ2'     :: CriticNet   -- ^ Target Critic θ2
                   , φOptim  :: T.Adam      -- ^ Policy Optimizer
                   , θ1Optim :: T.Adam      -- ^ Critic 1 Optimizer
                   , θ2Optim :: T.Adam      -- ^ Critic 2 Optimizer
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

-- | Save an Agent Checkpoint
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = do

        T.saveParams φ   (path ++ "/actorOnline.pt")
        T.saveParams φ'  (path ++ "/actorTarget.pt")
        T.saveParams θ1  (path ++ "/q1Online.pt")
        T.saveParams θ2  (path ++ "/q2Online.pt")
        T.saveParams θ1' (path ++ "/q1Target.pt")
        T.saveParams θ2' (path ++ "/q2Target.pt")

        saveOptim φOptim  (path ++ "/actorOptim")
        saveOptim θ1Optim (path ++ "/q1Optim")
        saveOptim θ2Optim (path ++ "/q2Optim")

        putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim iter actDim = do
        Agent{..} <- mkAgent obsDim actDim

        fφ    <- T.loadParams φ   (path ++ "/actor.pt")
        fφ'   <- T.loadParams φ'  (path ++ "/actor.pt")
        fθ1   <- T.loadParams θ1  (path ++ "/q1Online.pt")
        fθ2   <- T.loadParams θ2  (path ++ "/q2Online.pt")
        fθ1'  <- T.loadParams θ1' (path ++ "/q1Target.pt")
        fθ2'  <- T.loadParams θ2' (path ++ "/q2Target.pt")

        fφOpt  <- loadOptim iter β1 β2 (path ++ "/actorOptim")
        fθ1Opt <- loadOptim iter β1 β2 (path ++ "/q1Optim")
        fθ2Opt <- loadOptim iter β1 β2 (path ++ "/q2Optim")
       
        pure $ Agent fφ fφ' fθ1 fθ2 fθ1' fθ2' fφOpt fθ1Opt fθ2Opt

-- | Add Exploration Noise to Action
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
updateStep :: Int -> Int -> Agent -> Tracker -> ReplayBuffer T.Tensor -> IO Agent
updateStep _ 0 agent _ _ = pure agent
updateStep iteration epoch Agent{..} tracker buffer@ReplayBuffer{..} = do
    ε <- normal μ' σ'
    let a' = π φ' s' + ε
    y <- T.detach $ r + γ * q' θ1' θ2' s' a'

    let v1  = q θ1 s a
        v2  = q θ2 s a
        jQ1 = T.mseLoss v1 y
        jQ2 = T.mseLoss v2 y

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "\tΘ1 Loss:\t" ++ show jQ1
        putStrLn $ "\tΘ2 Loss:\t" ++ show jQ2
    _ <- trackLoss tracker iteration "Q1" (T.asValue jQ1 :: Float)
    _ <- trackLoss tracker iteration "Q2" (T.asValue jQ2 :: Float)

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηθ
    (θ2Online', θ2Optim') <- T.runStep θ2 θ1Optim jQ2 ηθ

    let updateActor :: IO (ActorNet, T.Adam)
        updateActor = do
            when (verbose && iteration `mod` 10 == 0) do
                putStrLn $ "\tφ  Loss:\t" ++ show jφ
            _ <- trackLoss tracker iteration "policy" (T.asValue jφ :: Float)
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

    (φOnline', φOptim') <- if iteration `mod` d == 0 
                              then updateActor
                              else pure (φ, φOptim)

    (φTarget', θ1Target', θ2Target') <- if iteration `mod` d == 0
                                           then syncTargets
                                           else pure (φ', θ1', θ2')

    let agent' = Agent φOnline' φTarget' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'           θ1Optim'  θ2Optim'

    updateStep iteration epoch' agent' tracker buffer
  where
    μ'     = T.zerosLike rpbActions
    σ'     = T.onesLike μ' * σ
    epoch' = epoch - 1
    s      = rpbStates
    a      = rpbActions
    r      = rpbRewards
    s'     = rpbStates'

-- | Perform Policy Update Steps
updatePolicy :: Int -> Agent -> Tracker -> ReplayBuffer T.Tensor -> Int -> IO Agent
updatePolicy iteration agent tracker buffer epochs = do
    memories <- bufferRandomSample batchSize buffer
    updateStep iteration epochs agent tracker memories

-- | Evaluate Policy
evaluatePolicy :: Int -> Int -> Agent -> HymURL -> Tracker -> T.Tensor 
               -> ReplayBuffer T.Tensor -> IO (ReplayBuffer T.Tensor, T.Tensor)
evaluatePolicy _ 0 _ _ _ states buffer = pure (buffer, states)
evaluatePolicy iteration step agent@Agent{..} envUrl tracker states buffer = do

    p       <- (iteration *) <$> numEnvsPool envUrl
    actions <- if p < warmupPeriode
                  then randomActionPool envUrl
                  else addNoise iteration (π φ states) >>= T.detach
    
    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions

    _ <- trackReward tracker iteration rewards
    when (iteration `mod` 8 == 0) do
        _ <- trackEnvState tracker envUrl iteration
        pure ()

    let keys   = head infos
    !states' <- if T.any dones 
                   then flip processGace keys <$> resetPool' envUrl dones
                   else pure $ processGace states'' keys

    let buffer' = bufferPush bufferSize buffer states actions rewards states' dones

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "\tEnvironments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t\t" ++ show de

    evaluatePolicy iteration step' agent envUrl tracker states' buffer'
  where
    step'   = step - 1

-- | Run Twin Delayed Deep Deterministic Policy Gradient Training
runAlgorithm :: Int -> Agent -> HymURL -> Tracker -> Bool -> ReplayBuffer T.Tensor
             -> T.Tensor -> IO Agent
runAlgorithm _ agent _ _ True _ _ = pure agent
runAlgorithm iteration agent envUrl tracker _ buffer states = do

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations

    (!memories', !states') <- evaluatePolicy iteration numSteps agent envUrl 
                                             tracker states buffer

    let buffer' = bufferPush' bufferSize buffer memories'

    !agent' <- if bufferLength buffer' < batchSize 
                  then pure agent
                  else updatePolicy iteration agent tracker buffer' numEpochs

    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . rpbRewards $ memories'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithm iteration' agent' envUrl tracker done' buffer' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ algorithm

-- | Train Twin Delayed Deep Deterministic Policy Gradient Agent on Environment
train :: Int -> Int -> HymURL -> TrackingURI -> IO Agent
train obsDim actDim envUrl trackingUri = do
    numEnvs <- numEnvsPool envUrl
    tracker <- mkTracker trackingUri algorithm >>= newRuns' numEnvs

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states = processGace states' keys
        buffer  = mkBuffer

    !agent <- mkAgent obsDim actDim >>= 
        (\agent' -> runAlgorithm 0 agent' envUrl tracker False buffer states)
    createModelArchiveDir algorithm >>= (`saveAgent` agent)

    endRuns' tracker

    pure agent

-- | Play Environment with Twin Delayed Deep Deterministic Policy Gradient Agent
--play :: Agent -> HymURL -> IO (M.Map String Float, Float)
