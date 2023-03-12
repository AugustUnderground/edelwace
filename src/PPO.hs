{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Proximal Policy Optimization Algorithm
module PPO ( algorithm
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
           , train
           , continue
           , play
           ) where

import Lib
import PPO.Defaults
import qualified RPB.MEM                          as MEM
import qualified Normal                           as D
import MLFlow           (TrackingURI)

import Control.Monad
import GHC.Generics
import qualified Torch                            as T
import qualified Torch.NN                         as NN
import qualified Torch.Distributions.Distribution as D
import qualified Torch.Distributions.Categorical  as D

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Actor Network Specification
data ActorNetSpec = ActorNetSpec { pObsDim :: Int, pActDim :: Int }
    deriving (Show, Eq)

-- | Critic Network Specification
newtype CriticNetSpec = CriticNetSpec { qObsDim :: Int }
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
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 256) 
                                             >>= weightInitNormal' )
                                       <*> ( T.sample (T.LinearSpec 256     256)
                                             >>= weightInitNormal' )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
                                             >>= weightInitNormal 0.0 wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec qObsDim 256) 
                                               >>= weightInitNormal' )
                                         <*> ( T.sample (T.LinearSpec 256 256) 
                                               >>= weightInitNormal' )
                                         <*> ( T.sample (T.LinearSpec 256 1) 
                                               >>= weightInitNormal' )

-- | Continuous Actor Network Forward Pass
πContinuous :: ActorNet -> T.Tensor -> T.Tensor
πContinuous ActorNet{..} o = a
  where
    a = T.tanh . T.linear pLayer2 
      . T.relu . T.linear pLayer1 
      . T.relu . T.linear pLayer0 
      $ o

-- | Discrete Actor Network Forward Pass
πDiscrete :: ActorNet -> T.Tensor -> T.Tensor
πDiscrete ActorNet{..} o = a
  where
    a = T.softmax (T.Dim 1) 
      . T.linear pLayer2 . T.tanh 
      . T.linear pLayer1 . T.tanh 
      . T.linear pLayer0 $ o

-- | Actor Network Forward Pass depending on `actionSpace`
π :: ActorNet -> T.Tensor -> T.Tensor
π = if actionSpace == Discrete then πDiscrete else πContinuous

-- | Critic Network Forward Pass
q :: CriticNet -> T.Tensor -> T.Tensor
q CriticNet{..} o = v
  where 
    v = T.linear qLayer2 . T.relu
      . T.linear qLayer1 . T.relu
      . T.linear qLayer0 $ o

------------------------------------------------------------------------------
-- PPO Agent
------------------------------------------------------------------------------

-- | PPO Agent
data Agent = Agent { φ      :: ActorNet             -- ^ Policy φ
                   , θ      :: CriticNet            -- ^ Critic θ
                   , logStd :: T.IndependentTensor  -- ^ Standard Deviation (Continuous)
                   , optim  :: T.Adam               -- ^ Joint Optimzier
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φ' <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim)

    logStd' <- if actionSpace == Discrete 
                  then T.makeIndependent  emptyTensor
                  else T.makeIndependent . toFloatGPU $ T.zeros' [1]

    let params = NN.flattenParameters φ' 
              ++ NN.flattenParameters θ' 
              ++ (if actionSpace == Discrete 
                     then []
                     else NN.flattenParameters [logStd'])
        optim' = T.mkAdam 0 β1 β2 params

    pure $ Agent φ' θ' logStd' optim'

-- | Save an Agent Checkpoint
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = do

        T.saveParams φ         (path ++ "/actor.pt")
        T.saveParams θ         (path ++ "/critic.pt")

        when (actionSpace == Discrete) do
            T.save [logStd'] (path ++ "/logStd.pt")

        saveOptim optim        (path ++ "/optim")

        putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "
  where
    logStd' = T.toDependent logStd

-- | Save an Agent and return the agent
saveAgent' :: String -> Agent -> IO Agent
saveAgent' p a = saveAgent p a >> pure a

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim actDim iter = do
        Agent{..} <- mkAgent obsDim actDim

        fφ      <- T.loadParams φ   (path ++ "/actor.pt")
        fθ      <- T.loadParams θ   (path ++ "/critic.pt")

        flogStd <- if actionSpace == Discrete 
                      then T.makeIndependent emptyTensor
                      else T.load (path ++ "/logStd.pt") 
                                >>= T.makeIndependent . head

        fopt     <- loadOptim iter β1 β2 (path ++ "/optim")
       
        pure $ Agent fφ fθ flogStd fopt

-- | Get value and distribution
actContinous :: Agent -> T.Tensor -> (D.Normal, T.Tensor)
actContinous Agent{..} s = (dist, value)
  where
    value = q θ s
    μ  = π φ s
    σ' = T.exp . T.toDependent $ logStd
    σ  = T.expand σ' True (T.shape μ)
    dist = D.Normal μ σ

-- | Get value and distribution without grad
actContinous' :: Agent -> T.Tensor -> IO (D.Normal, T.Tensor)
actContinous' Agent{..} s = do
    value <- T.detach $ q θ s
    μ  <- T.detach $ π φ s
    σ' <- T.detach $ T.exp . T.toDependent $ logStd
    let σ = T.expand σ' True (T.shape μ)
        dist = D.Normal μ σ
    pure (dist, value)

-- | Get value and distribution
actDiscrete :: Agent -> T.Tensor -> (D.Categorical, T.Tensor)
actDiscrete Agent{..} s = (dist, value)
  where
    value = q θ s
    probs = π φ s
    dist  = D.fromProbs probs

-- | Get value and distribution without grad
actDiscrete' :: Agent -> T.Tensor -> IO (D.Categorical, T.Tensor)
actDiscrete' Agent{..} s = do
    value <- T.detach $ q θ s
    probs <- T.detach $ π φ s
    let dist = D.fromProbs probs
    pure (dist, value)

-- | Get entropy, logprobs and values for training
act :: ActionSpace -> Agent -> T.Tensor -> T.Tensor
    -> (T.Tensor, T.Tensor, T.Tensor)
act Discrete agent states actions = (entropy, logProbs, values)
  where
    (dist, values) = actDiscrete agent states
    entropy        = T.reshape [-1,1] $ D.entropy dist
    logProbs       = T.reshape [-1,1] $ D.logProb dist actions
act Continuous agent states actions = (entropy, logProbs, values)
  where
    (dist, values) = actContinous agent states
    entropy        = D.entropy dist
    logProbs       = D.logProb dist actions

-- | Get actions, logprobs and values for evaluation
act' :: ActionSpace -> Agent -> T.Tensor
     -> IO (T.Tensor, T.Tensor, T.Tensor)
act' Discrete agent states = do
    (dist, values) <- actDiscrete' agent states
    actions        <- T.reshape [-1,1] <$> D.sample dist [1]
    let logProbs    = T.reshape [-1,1]  $ D.logProb dist actions
    pure (actions, logProbs, values)
act' Continuous agent states = do
    (dist, values) <- actContinous' agent states
    actions        <- T.clamp (- 1.0) 1.0 <$> D.sample dist []
    let logProbs   = D.logProb dist actions
    pure (actions, logProbs, values)

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | No need to update / backprop logStd for discrete Agent
updateAgent :: ActionSpace -> Agent -> T.Tensor -> IO Agent
updateAgent Discrete Agent{..} loss = do
    ((φ', θ'), optim') <- T.runStep (φ, θ) optim loss η
    pure $ Agent φ' θ' logStd optim'
updateAgent Continuous Agent{..} loss = do
    ((φ', θ', logStd'), optim') <- T.runStep (φ, θ, logStd) optim loss η
    pure $ Agent φ' θ' logStd' optim'

-- | Policy Update Step
updateStep :: Agent -> MEM.Loader T.Tensor -> IO (Agent, T.Tensor)
updateStep agent MEM.Loader{..} = do
    agent' <- updateAgent actionSpace agent loss
    pure (agent', loss)
  where
    (entropies, logProbs, values) 
             = act actionSpace agent states' actions'
    ratios   = T.exp $ logProbs - logProbs'
    surr1    = ratios * advantages'
    surr2    = advantages' * T.clamp (1.0 - ε) (1.0 + ε) ratios
    πLoss    = T.mean . fst . T.minDim (T.Dim 1) T.KeepDim
             $ T.cat (T.Dim 1) [surr1, surr2]
    rewards  = scaleRewards returns' rewardScale
    qLoss    = T.mseLoss values rewards
    --qLoss    = T.mseLoss values MEM.returns'
    loss     = T.mean $ (- πLoss) + 0.5 * qLoss - δ * entropies
 
-- | Run Policy Update
updatePolicy :: Int -> Int -> Agent -> Tracker -> MEM.Loader [T.Tensor] 
             -> T.Tensor -> IO Agent
updatePolicy iteration epoch agent tracker (MEM.Loader [] [] [] [] []) loss = do
    _ <- trackLoss tracker (iteration * numEpochs + epoch) 
                   "policy" (T.asValue loss :: Float)
    when (verbose && epoch `mod` 4 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch ++ " Loss:\t" ++ show loss
    pure agent
updatePolicy iteration epoch agent tracker loader _ = do
    (agent', loss') <- updateStep agent batch
    updatePolicy iteration epoch agent' tracker loader' loss'
  where
    batch   = head <$> loader
    loader' = tail <$> loader

-- | Evaluation Step
evaluateStep :: Int -> Int -> Agent -> HymURL -> Tracker -> T.Tensor 
             -> MEM.Buffer T.Tensor -> IO (MEM.Buffer T.Tensor, T.Tensor)
evaluateStep _ 0 _ _ _ states mem = do
    when verbose do
        let tot = T.sumAll . MEM.rewards $ mem
        putStrLn $ "\tStep " ++ show numSteps ++ "\tTotal Reward:\t" ++ show tot
    pure (mem, states)
evaluateStep iteration step agent envUrl tracker states mem = do
    (actions', logProbs', values')         <- act' actionSpace agent states

    (!states'', !rewards', !dones, !infos) <- if actionSpace == Discrete
                                                 then stepPool' envUrl actions'
                                                 else stepPool  envUrl actions'

    _ <- trackReward tracker (iteration * numSteps + (numSteps - step)) rewards'
    when (even step) do
        _ <- trackEnvState tracker envUrl (iteration * numSteps + (numSteps - step))
        pure ()

    when (verbose && step `mod` 10 == 0) do
        let men = T.mean rewards'
        putStrLn $ "\tStep " ++ show (numSteps - step) ++ " / " ++ show numSteps 
                             ++ ":\n\t\tAverage Reward:\t" ++ show men

    let keys    = head infos
    !states' <- if T.any dones 
                   then flip processGace keys <$> resetPool' envUrl dones
                   else pure $ processGace states'' keys

    let masks' = T.logicalNot dones
        mem'   = MEM.push mem states' actions' logProbs' rewards' values' masks'

    evaluateStep iteration step' agent envUrl tracker states' mem'
  where
    step' = step - 1

-- | Evaluate Current Policy
evaluatePolicy :: Int -> Agent -> HymURL -> Tracker -> T.Tensor 
               -> IO (MEM.Buffer T.Tensor, T.Tensor)
evaluatePolicy iteration agent envUrl tracker states = do
    evaluateStep iteration numSteps agent envUrl tracker states mem
  where
    mem = MEM.mkBuffer

-- | Run Proximal Porlicy Optimization Training
runAlgorithm :: Int -> Agent -> HymURL -> Tracker -> Bool -> T.Tensor -> IO Agent
runAlgorithm _ agent _ _ True _ = pure agent
runAlgorithm iteration agent envUrl tracker _ states = do

    when verbose do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!mem', !states') <- evaluatePolicy iteration agent envUrl tracker states

    let !loader = MEM.mkLoader mem' batchSize γ τ

    agent' <- T.foldLoop agent numEpochs 
                (\gnt epc -> updatePolicy iteration epc gnt tracker loader emptyTensor)

    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . MEM.rewards $ mem'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithm iteration' agent' envUrl tracker done' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ show algorithm

-- | Train Proximal Policy Optimization Agent on Environment
train :: Int -> Int -> HymURL -> TrackingURI -> IO Agent
train obsDim actDim envUrl trackingUri = do
    numEnvs <- numEnvsPool envUrl
    tracker <- mkTracker trackingUri (show algorithm) >>= newRuns' numEnvs

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states = processGace states' keys

    !agent <- mkAgent obsDim actDim >>= 
        (\agent' -> runAlgorithm 0 agent' envUrl tracker False states )
    createModelArchiveDir (show algorithm) >>= (`saveAgent` agent)
    
    endRuns' tracker

    pure agent

-- | Continue training with Twin Delayed Deep Deterministic Policy Gradient Agent
continue :: HymURL -> TrackingURI -> Agent -> IO Agent
continue envUrl trackingUri agent = do
    numEnvs <- numEnvsPool envUrl
    tracker <- mkTracker trackingUri expName >>= newRuns' numEnvs

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states = processGace states' keys

    !agent' <- runAlgorithm 0 agent envUrl tracker False states

    endRuns' tracker
    pure agent'
  where
    expName = show algorithm ++ "-" 
            ++ (reverse . takeWhile (/= '/') . reverse $ envUrl)
            ++ "-cont"

-- | Play Environment with Proximal Policy Optimization Agent
play :: HymURL -> TrackingURI -> Agent -> IO ()
play _ _ _ = do
    pure ()
