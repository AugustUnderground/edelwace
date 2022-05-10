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
           , updatePolicy
           -- , play
           ) where

import Lib
import TD3.Defaults
import RPB
import qualified RPB.RPB                   as RPB
import qualified RPB.HER                   as HER
import MLFlow               (TrackingURI)

import Control.Monad
import GHC.Generics
import qualified Data.Set                  as S
import qualified Data.Map                  as M
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (negative)
import qualified Torch.NN                  as NN

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
data CriticNet = CriticNet { q1Layer0 :: T.Linear
                           , q1Layer1 :: T.Linear
                           , q1Layer2 :: T.Linear 
                           , q2Layer0 :: T.Linear
                           , q2Layer1 :: T.Linear
                           , q2Layer2 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet 
                           <$> T.sample   (T.LinearSpec pObsDim 128) 
                           <*> T.sample   (T.LinearSpec 128     128)
                           <*> ( T.sample (T.LinearSpec 128 pActDim)
                                    >>= weightInitUniform (- wInit) wInit )
                                    -- >>= weightInitNormal (-wInit) wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet 
                            <$> T.sample   (T.LinearSpec dim 128) 
                            <*> T.sample   (T.LinearSpec 128 128) 
                            <*> ( T.sample (T.LinearSpec 128 1) 
                                    >>= weightInitUniform (- wInit) wInit )
                            <*> T.sample   (T.LinearSpec dim 128) 
                            <*> T.sample   (T.LinearSpec 128 128) 
                            <*> ( T.sample (T.LinearSpec 128 1) 
                                    >>= weightInitUniform (- wInit) wInit )
        where dim = qObsDim + qActDim

--instance T.Randomizable CriticNetSpec CriticNet where
--    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec qObsDim 128) 
--                                               >>= weightInitUniform' )
--                                         <*> ( T.sample (T.LinearSpec dim     256) 
--                                               >>= weightInitUniform' )
--                                         <*> ( T.sample (T.LinearSpec 256     1) 
--                                               >>= weightInitUniform' )
--                                         <*> ( T.sample (T.LinearSpec qObsDim 128) 
--                                               >>= weightInitUniform' )
--                                         <*> ( T.sample (T.LinearSpec dim     256) 
--                                               >>= weightInitUniform' )
--                                         <*> ( T.sample (T.LinearSpec 256     1)
--                                               >>= weightInitUniform' )
--        where dim = 128 + qActDim

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> T.Tensor
π ActorNet{..} o = a
  where
    a = T.tanh . T.linear pLayer2
      . T.relu . T.linear pLayer1
      . T.relu . T.linear pLayer0
      $ o

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor -> (T.Tensor, T.Tensor)
q CriticNet{..} o a = (v1,v2)
  where 
    x  = T.cat (T.Dim $ -1) [o,a]
    v1 = T.linear q1Layer2 . T.relu
       . T.linear q1Layer1 . T.relu
       . T.linear q1Layer0 $ x
    v2 = T.linear q2Layer2 . T.relu
       . T.linear q2Layer1 . T.relu
       . T.linear q2Layer0 $ x
--q :: CriticNet -> T.Tensor -> T.Tensor -> (T.Tensor, T.Tensor)
--q CriticNet{..} o a = (v1,v2)
--  where 
--    o1 = T.leakyRelu negativeSlope $ T.linear q1Layer0 o
--    o2 = T.leakyRelu negativeSlope $ T.linear q2Layer0 o
--    x1 = T.cat (T.Dim $ -1) [o1, a]
--    x2 = T.cat (T.Dim $ -1) [o2, a]
--    v1 = T.linear q1Layer2 . T.leakyRelu negativeSlope . T.linear q1Layer1 $ x1
--    v2 = T.linear q2Layer2 . T.leakyRelu negativeSlope . T.linear q2Layer1 $ x2

-- | Convenience Function, takes the minimum of both online actors
q' :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
q' cn o a = fst . T.minDim (T.Dim 1) T.KeepDim . T.cat (T.Dim 1) $ [v1,v2]
  where
    (v1,v2) = q cn o a
-- q' = ((fst .) .) . ((T.minDim (T.Dim 1) T.KeepDim .) .) . ((T.cat (T.Dim 1) . ) .) . q

------------------------------------------------------------------------------
-- TD3 Agent
------------------------------------------------------------------------------

-- | TD3 Agent
data Agent = Agent { φ      :: ActorNet    -- ^ Online Policy φ
                   , φ'     :: ActorNet    -- ^ Target Policy φ'
                   , θ      :: CriticNet   -- ^ Online Critic θ
                   , θ'     :: CriticNet   -- ^ Target Critic θ
                   , φOptim :: T.Adam      -- ^ Policy Optimizer
                   , θOptim :: T.Adam      -- ^ Critic Optimizer
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φOnline  <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    φTarget' <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θOnline  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θTarget' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)

    let φTarget = copySync φTarget' φOnline
        θTarget = copySync θTarget' θOnline
        φOpt    = T.mkAdam 0 β1 β2 (NN.flattenParameters φOnline)
        θOpt    = T.mkAdam 0 β1 β2 (NN.flattenParameters θOnline)

    pure $ Agent φOnline φTarget θOnline θTarget φOpt θOpt

-- | Save an Agent Checkpoint
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = do

        T.saveParams φ  (path ++ "/actorOnline.pt")
        T.saveParams φ' (path ++ "/actorTarget.pt")
        T.saveParams θ  (path ++ "/q1Online.pt")
        T.saveParams θ' (path ++ "/q1Target.pt")

        saveOptim φOptim (path ++ "/actorOptim")
        saveOptim θOptim (path ++ "/q1Optim")

        putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "

-- | Save an Agent and return the agent
saveAgent' :: String -> Agent -> IO Agent
saveAgent' p a = saveAgent p a >> pure a

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim iter actDim = do
        Agent{..} <- mkAgent obsDim actDim

        fφ    <- T.loadParams φ  (path ++ "/actor.pt")
        fφ'   <- T.loadParams φ' (path ++ "/actor.pt")
        fθ    <- T.loadParams θ  (path ++ "/q1Online.pt")
        fθ'   <- T.loadParams θ' (path ++ "/q1Target.pt")
        fφOpt <- loadOptim iter β1 β2 (path ++ "/actorOptim")
        fθOpt <- loadOptim iter β1 β2 (path ++ "/q1Optim")
       
        pure $ Agent fφ fφ' fθ fθ' fφOpt fθOpt

-- | Get action from online policy with naive / static Exploration Noise
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} s = do
    ε <- toFloatGPU <$> T.randnLikeIO a
    pure $ T.clamp actionLow actionHigh (a + (ε * σAct))
  where
    a = π φ s

-- | Get action from online policy with dynamic Exploration Noise
act' :: Int -> Agent -> T.Tensor -> IO T.Tensor
act' t Agent{..} s = do
    ε <- toFloatGPU <$> T.normalIO μ σ
    pure $ T.clamp actionLow actionHigh (a + ε)
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
    ε' <- toFloatGPU <$> T.randnLikeIO a 
    let ε = T.clamp (- c) c (ε' * σEval)
    pure $ T.clamp actionLow actionHigh (a + ε)
  where
    a = π φ' s

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Int -> Int -> Agent -> Tracker -> RPB.Buffer T.Tensor -> IO Agent
updateStep iteration epoch agent@Agent{..} tracker RPB.Buffer{..} = do
    a' <- evaluate agent s' >>= T.detach
    v' <- T.detach $ q' θ' s' a'
    y  <- T.detach $ r + ((1.0 - d') * γ * v')

    let (v1,v2) = q θ s a
        jQ = T.mseLoss v1 y + T.mseLoss v2 y

    (θOnline', θOptim') <- T.runStep θ θOptim jQ ηθ

    when (verbose && epoch `mod` 10 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch
        putStrLn $ "\t\tΘ Loss:\t" ++ show jQ
    _ <- trackLoss tracker (iter' !! epoch') "Critic_Loss" (T.asValue jQ :: Float)

    (φOnline', φOptim') <- if epoch `mod` d == 0 
                              then updateActor
                              else pure (φ, φOptim)

    (φTarget', θTarget') <- if epoch' == 0
                               then syncTargets
                               else pure (φ', θ')

    pure $ Agent φOnline' φTarget' θOnline' θTarget' φOptim' θOptim'
  where
    iter'  = reverse [(iteration * numEpochs) .. (iteration * numEpochs + numEpochs)]
    epoch' = epoch - 1
    s      = states
    a      = actions
    r      = rewards
    d'     = dones
    s'     = states'
    updateActor :: IO (ActorNet, T.Adam)
    updateActor = do
        when (verbose && epoch `mod` 10 == 0) do
            putStrLn $ "\t\tφ Loss:\t" ++ show jφ
        _ <- trackLoss tracker ((iter' !! epoch') `div` d)
                       "Actor_Loss" (T.asValue jφ :: Float)
        T.runStep φ φOptim jφ ηφ
      where
        (v,_) = q θ s $ π φ s
        jφ    = T.negative . T.mean $ v
    syncTargets :: IO (ActorNet, CriticNet)
    syncTargets = do
        when verbose do
            putStrLn "\t\tUpdating Targets."
        φTarget' <- softSync τ φ' φ
        θTarget' <- softSync τ θ' θ
        pure (φTarget', θTarget')

-- | Perform Policy Update Steps
updatePolicy :: Int -> Agent -> Tracker -> [RPB.Buffer T.Tensor] -> IO Agent
updatePolicy _         agent _       []              = pure agent
updatePolicy iteration agent tracker (batch:batches) = do
    agent' <- updateStep iteration epochs agent tracker batch
    updatePolicy iteration agent' tracker batches
  where
    epochs = length batches

-- | Evaluate Policy for usually just one step and a pre-determined warmup Period
evaluatePolicyRPB :: Int -> Int -> Agent -> HymURL -> Tracker -> T.Tensor 
                  -> RPB.Buffer T.Tensor -> IO (RPB.Buffer T.Tensor, T.Tensor)
evaluatePolicyRPB _ 0 _ _ _ states buffer = pure (buffer, states)
evaluatePolicyRPB iteration step agent envUrl tracker states buffer = do

    p       <- (iteration *) <$> numEnvsPool envUrl
    actions <- if p < warmupPeriode
                  then randomActionPool envUrl
                  else act agent states >>= T.detach
    
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

    scaler  <- head . M.elems <$> acePoolScaler envUrl
    keys    <- infoPool envUrl

    actions <- if iteration `mod` randomEpisode == 0 -- iteration <= 0 
                  then nanToNum' <$> randomActionPool envUrl
                  else act agent (toFloatGPU $ T.cat (T.Dim 1) [states, targets])
                            >>= (T.detach . toFloatCPU)

    (!states', !targets', !targets'', !rewards, !dones) 
            <- postProcess' scaler <$> stepPool envUrl actions

    let dones'  = T.reshape [-1] . T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        done'   = S.union done . S.fromList $ (T.asValue dones' :: [Int])
        success = (realToFrac . S.size $ done') / realToFrac numEnvs
        buffer' = HER.push'' bufferSize buffer states actions rewards
                             states' dones targets' targets'' 

    when (step < numSteps) do
        _   <- trackLoss tracker (iter' !! step) "Success" success
        pure ()

    _       <- trackReward tracker (iter' !! step) rewards

    when (even step) do
        _   <- trackEnvState tracker envUrl (iter' !! step)
        pure ()

    when (verbose && step `mod` 10 == 0) do
        putStrLn $ "\tStep " ++ show step ++ ", Average Success: \t" 
                             ++ show (100.0 * success) ++ "%"

    (states''', targets''', _) 
            <- if T.any dones && (step' < numSteps)
                  then postProcess keys scaler <$> resetPool' envUrl dones
                  else pure (states', targets', targets'')

    evaluatePolicyHER iteration step' done' numEnvs agent envUrl tracker 
                      states''' targets''' buffer'
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
    batch <- fmap toFloatGPU <$> RPB.sampleIO batchSize buffer'

    !agent' <- if RPB.size buffer' < batchSize 
                  then pure agent
                  else updateStep iteration numEpochs agent tracker batch

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
        let eve = if iteration `mod` randomEpisode == 0
                    then "Random Exploration"
                    else "Policy Exploitation"
        putStrLn $ "Episode " ++ show iteration ++ " / " ++ show numIterations
                ++ ": " ++ eve

    numEnvs       <- numEnvsPool envUrl
    trajectories  <- evaluatePolicyHER iteration 0 S.empty numEnvs agent envUrl 
                                       tracker states targets HER.empty

    let episodes = concatMap HER.epsSplit $ HER.envSplit numEnvs trajectories

    predicate <- HER.targetCriterion . head . M.elems <$> acePoolPredicate envUrl

    buffer' <- if strategy == HER.Random
                  then HER.sampleTargets strategy k predicate $ 
                             foldl (HER.push' bufferSize) buffer episodes
                  else foldM (\b b' -> HER.push' bufferSize b 
                                   <$> HER.sampleTargets strategy k predicate b') 
                             buffer episodes

    when verbose do
        putStrLn $ "\tEpisode Buffer length:\t\t" ++ show (HER.size trajectories)
        putStrLn $ "\tReplay Buffer Size:\t\t" ++ show (HER.size buffer')
        putStrLn $ "\tSuccesses before augmentation:\t" 
                 ++ ( show . head . T.shape . T.nonzero 
                    $ (1.0 + HER.rewards trajectories))
        putStrLn $ "\tSuccesses after augmentation:\t" 
                 ++ ( show . head . T.shape . T.nonzero 
                    $ (1.0 + HER.rewards buffer'))

    batches <- RPB.randomBatches batchSize numEpochs $ HER.asRPB buffer'
    !agent' <- if HER.size buffer' <= batchSize 
                  then pure agent
                  else updatePolicy iteration agent tracker batches

    saveAgent ptPath agent

    keys   <- infoPool envUrl
    scaler <- head . M.elems <$> acePoolScaler envUrl
    (states', targets', _) <- postProcess keys scaler  <$> resetPool envUrl

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
    states' <- toFloatCPU <$> resetPool envUrl
    keys    <- infoPool envUrl
    scaler <- head . M.elems <$> acePoolScaler envUrl
    let (states, targets, _) = postProcess keys scaler states'
    runAlgorithmHER 0 agent envUrl tracker False HER.empty targets states 
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
