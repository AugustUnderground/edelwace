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
           , saveAgent'
           , loadAgent
           , π
           , q
           , q'
           , act
           , act'
           , evaluate
           , train
           -- , play
           ) where

import Lib
import TD3.Defaults
import RPB
import qualified RPB.RPB  as RPB
import qualified RPB.HER  as HER
import MLFlow       (TrackingURI)

import Control.Monad
import GHC.Generics
import qualified Data.Set as S
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
                         , pLayer3 :: T.Linear 
                         } deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { qLayer0 :: T.Linear
                           , qLayer1 :: T.Linear
                           , qLayer2 :: T.Linear 
                           , qLayer3 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 64) 
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 64      64)
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 64      64)
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 64 pActDim)
                                             >>= weightInitUniform (-wInit) wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec dim 64) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 64  64) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 64  64) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 64  1) 
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
q' c1 c2 o a = v
  where
    q1 = q c1 o a
    q2 = q c2 o a
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

-- | Save an Agent and return the agent
saveAgent' :: String -> Agent -> IO Agent
saveAgent' p a = saveAgent p a >> pure a

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

-- | Add Dynamic Exploration Noise to Action based on #Episode
-- addNoise :: Int -> T.Tensor -> IO T.Tensor
-- addNoise t action = do
--     ε <- normal' [n, 1]
--     pure $ T.clamp actionLow actionHigh (action + (σ' * ε))
--   where
--     -- n  = T.shape action !! 1
--     n  = head $ T.shape action
--     d' = realToFrac decayPeriod
--     t' = realToFrac t
--     m  = min 1.0 (t' / d')
--     σ' = toTensor $ σMax - (σMax - σMin) * m

-- | Get action from online policy with naive / static Exploration Noise
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} s = do
    ε <- toFloatGPU <$> T.normalIO μ σ
    T.detach $ T.clamp actionLow actionHigh (a + ε)
  where
    a = π φ s
    μ = toFloatGPU $ T.zerosLike a
    σ = T.repeat (T.shape a) σAct

-- | Get action from online policy with dynamic Exploration Noise
act' :: Int -> Agent -> T.Tensor -> IO T.Tensor
act' t Agent{..} s = do
    ε <- toFloatGPU <$> T.normalIO μ σ
    T.detach $ T.clamp actionLow actionHigh (a + ε)
  where
    a  = π φ s
    d' = realToFrac decayPeriod
    t' = realToFrac t
    m  = min 1.0 (t' / d')
    μ  = toFloatGPU $ T.zerosLike a
    σ = T.repeat (T.shape a) . toTensor $ σMax - (σMax - σMin) * m

-- | Get an action
evaluate :: Agent -> T.Tensor -> IO T.Tensor
evaluate Agent{..} s = do
    ε <- toFloatGPU . T.clamp (- c) c <$> T.normalIO μ σ
    T.detach (a + ε)
  where
    a = π φ' s
    μ = toFloatGPU $ T.zerosLike a
    σ = T.repeat (T.shape a) σEval

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Int -> Int -> Agent -> Tracker -> RPB.Buffer T.Tensor 
           -> IO Agent
updateStep _ 0 agent _ _ = pure agent
updateStep iteration epoch agent@Agent{..} tracker buffer@RPB.Buffer{..} = do
    a' <- evaluate agent s'
    let v' = q' θ1' θ2' s' a'
    y <- T.detach $ r + (d' * γ * v')

    let v1  = q θ1 s a
        v2  = q θ2 s a
        jQ1 = T.mseLoss v1 y
        jQ2 = T.mseLoss v2 y

    when (verbose && epoch `mod` 10 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch
        putStrLn $ "\t\tΘ1 Loss:\t" ++ show jQ1
        putStrLn $ "\t\tΘ2 Loss:\t" ++ show jQ2
    _ <- trackLoss tracker (iter' !! epoch') "Q1" (T.asValue jQ1 :: Float)
    _ <- trackLoss tracker (iter' !! epoch') "Q2" (T.asValue jQ2 :: Float)
    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηθ
    (θ2Online', θ2Optim') <- T.runStep θ2 θ1Optim jQ2 ηθ

    let updateActor :: IO (ActorNet, T.Adam)
        updateActor = do
            when (verbose && epoch `mod` 10 == 0) do
                putStrLn $ "\t\tφ  Loss:\t" ++ show jφ
            _ <- trackLoss tracker (iter' !! epoch') "policy" (T.asValue jφ :: Float)
            T.runStep φ φOptim jφ ηφ
          where
            a'' = π φ s
            v   = q θ1 s a''
            jφ  = negate . T.mean $ v
        syncTargets :: IO (ActorNet, CriticNet, CriticNet)
        syncTargets = do
            φTarget'  <- softSync τ φ'  φ
            θ1Target' <- softSync τ θ1' θ1 
            θ2Target' <- softSync τ θ2' θ2 
            pure (φTarget', θ1Target', θ2Target')

    (φOnline', φOptim') <- if epoch `mod` dPolicy == 0 
                              then updateActor
                              else pure (φ, φOptim)

    (φTarget', θ1Target', θ2Target') <- if iteration `mod` dTarget == 0
                                           then syncTargets
                                           else pure (φ', θ1', θ2')

    let agent' = Agent φOnline' φTarget' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'           θ1Optim'  θ2Optim'

    updateStep iteration epoch' agent' tracker buffer
  where
    iter'  = reverse [(iteration * numEpochs) .. (iteration * numEpochs + numEpochs)]
    epoch' = epoch - 1
    s      = states
    a      = actions
    r      = rewards
    d'     = 1.0 - dones
    s'     = states'

-- | Perform Policy Update Steps
updatePolicy :: Int -> Agent -> Tracker -> RPB.Buffer T.Tensor -> Int 
             -> IO Agent
updatePolicy iteration agent tracker buffer epochs = do
    memories <- RPB.sampleIO batchSize buffer
    updateStep iteration epochs agent tracker memories

-- | Evaluate Policy for usually just one step and a pre-determined warmup Period
evaluatePolicyRPB :: Int -> Int -> Agent -> HymURL -> Tracker -> T.Tensor 
               -> RPB.Buffer T.Tensor -> IO (RPB.Buffer T.Tensor, T.Tensor)
evaluatePolicyRPB _ 0 _ _ _ states buffer = pure (buffer, states)
evaluatePolicyRPB iteration step agent envUrl tracker states buffer = do

    p       <- (iteration *) <$> numEnvsPool envUrl
    actions <- if p < warmupPeriode
                  then randomActionPool envUrl
                  else act' iteration agent states >>= T.detach
                  -- else addNoise iteration (π φ states) >>= T.detach
    
    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions

    _ <- trackReward tracker iteration rewards
    when (iteration `mod` 4 == 0) do
        _ <- trackEnvState tracker envUrl iteration
        pure ()

    let keys   = head infos
    !states' <- if T.any dones 
                   then flip processGace keys <$> resetPool' envUrl dones
                   else pure $ processGace states'' keys

    let buffer' = RPB.push bufferSize buffer states actions rewards states' dones

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "\tAverage Reward: \t" ++ show (T.mean rewards)

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "\tEnvironments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t\t" ++ show de

    evaluatePolicyRPB iteration step' agent envUrl tracker states' buffer'
  where
    step'   = step - 1

-- | Evaluate Policy until all envs are done at least once
evaluatePolicyHER :: Int -> Int -> S.Set Int -> Int -> Agent -> HymURL 
                  -> Tracker -> T.Tensor -> T.Tensor -> HER.Buffer T.Tensor 
                  -> IO (HER.Buffer T.Tensor)
evaluatePolicyHER iteration step done numEnvs agent envUrl tracker states 
                  targets buffer | S.size done == numEnvs = pure buffer
                                 | otherwise              = do

    -- actions <- forM [states, targets] T.clone
    --             >>= (addNoise iteration . π φ . T.cat (T.Dim 1))
    --             >>= T.detach

    actions <- forM [states, targets] T.clone 
                >>= act' iteration agent . T.cat (T.Dim 1) 
                >>= T.detach

    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions

    let dones' = T.reshape [-1] . T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        done'  = S.union done . S.fromList $ (T.asValue dones' :: [Int])
        keys   = head infos

    (states', targets', targets'') <- if T.any dones 
           then flip   processGace'' keys <$> resetPool' envUrl dones
           else pure $ processGace'' states'' keys

    let buffer' = HER.push bufferSize buffer states actions rewards states'
                           dones targets' targets''

    _ <- trackReward tracker (iter' !! step) rewards
    when (even step) do
        _ <- trackEnvState tracker envUrl (iter' !! step)
        pure ()

    when (verbose && step `mod` 10 == 0) do
        putStrLn $ "\tStep " ++ show step ++ ", Average Reward: \t" 
                             ++ show (T.mean rewards)

    evaluatePolicyHER iteration step' done' numEnvs agent envUrl tracker 
                      states' targets' buffer'
  where
    step' = step + 1
    iter' = [(iteration * numSteps) .. (iteration * numSteps + numSteps)]

-- | Run Twin Delayed Deep Deterministic Policy Gradient Training with RPB
runAlgorithmRPB :: Int -> Agent -> HymURL -> Tracker -> Bool 
                -> RPB.Buffer T.Tensor -> T.Tensor -> IO Agent
runAlgorithmRPB _ agent _ _ True _ _ = pure agent
runAlgorithmRPB iteration agent envUrl tracker _ buffer states = do

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations

    (!memories', !states') <- evaluatePolicyRPB iteration numSteps agent envUrl 
                                                tracker states buffer

    let buffer' = RPB.push' bufferSize buffer memories'

    !agent' <- if RPB.size buffer' < batchSize 
                  then pure agent
                  else updatePolicy iteration agent tracker buffer' numEpochs

    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . RPB.rewards $ memories'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithmRPB iteration' agent' envUrl tracker done' buffer' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ show algorithm

-- | Run Twin Delayed Deep Deterministic Policy Gradient Training with HER
runAlgorithmHER :: Int -> Agent -> HymURL -> Tracker -> Bool 
                -> HER.Buffer T.Tensor -> T.Tensor -> T.Tensor -> IO Agent
runAlgorithmHER _ agent _ _ True _ _ _ = pure agent
runAlgorithmHER iteration agent envUrl tracker _ buffer targets states = do

    when verbose do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations

    numEnvs       <- numEnvsPool envUrl
    trajectories  <- evaluatePolicyHER iteration 0 S.empty numEnvs agent envUrl 
                                       tracker states targets HER.empty

    let episodes = concatMap HER.epsSplit $ HER.envSplit numEnvs trajectories

    buffer' <- if strategy == HER.Random
                  then HER.sampleTargets strategy k relTol $ 
                        foldl (HER.push' bufferSize) buffer episodes
                  else foldM (\b b' -> HER.push' bufferSize b 
                                   <$> HER.sampleTargets strategy k relTol b') 
                             buffer episodes

    let memory = HER.asRPB buffer'

    !agent' <- updatePolicy iteration agent tracker memory numEpochs
    saveAgent ptPath agent 

    keys <- infoPool envUrl
    (states', targets', _) <- flip processGace'' keys <$> resetPool envUrl

    runAlgorithmHER iteration' agent' envUrl tracker done' buffer' targets' states' 
  where
    done'         = iteration >= numIterations
    iteration'    = iteration + 1
    ptPath        = "./models/" ++ show algorithm

-- | Handle training for different replay buffer types
train' :: HymURL -> Tracker -> BufferType -> Agent -> IO Agent
train' envUrl tracker RPB agent = do
    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl
    let !states = processGace states' keys
    runAlgorithmRPB 0 agent envUrl tracker False RPB.mkBuffer states 
train' envUrl tracker HER agent = do
    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl
    let (states, targets, _) = processGace'' states' keys
    runAlgorithmHER 0 agent envUrl tracker False HER.mkBuffer targets states 
train' _ _ _ _ = undefined

-- | Train Twin Delayed Deep Deterministic Policy Gradient Agent on Environment
train :: Int -> Int -> HymURL -> TrackingURI -> IO Agent
train obsDim actDim envUrl trackingUri = do
    numEnvs <- numEnvsPool envUrl
    tracker <- mkTracker trackingUri (show algorithm) >>= newRuns' numEnvs

    !agent <- mkAgent obsDim actDim >>= train' envUrl tracker bufferType
    createModelArchiveDir (show algorithm) >>= (`saveAgent` agent)

    endRuns' tracker

    pure agent

-- | Play Environment with Twin Delayed Deep Deterministic Policy Gradient Agent
--play :: Agent -> HymURL -> IO (M.Map String Float, Float)
