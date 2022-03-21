{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Proximal Policy Optimization Algorithm
module PPO ( algorithm
           , Agent (..)
           , mkAgent
           , saveAgent
           , π
           , q
           , train
           -- , play
           ) where

import Lib
import RPB
import PPO.Defaults

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
newtype CriticNetSpec = CriticNetSpec { qObsDim :: Int }
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
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 256) 
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256     256)
                                             >>= weightInit' )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
                                             >>= weightInit wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec qObsDim 256) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 256 256) 
                                               >>= weightInit' )
                                         <*> ( T.sample (T.LinearSpec 256 1) 
                                               >>= weightInit' )

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> T.Tensor
π ActorNet{..} o = a
  where
    a = T.tanh . T.linear pLayer2 
      . T.relu . T.linear pLayer1 
      . T.relu . T.linear pLayer0 
      $ o

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
data Agent = Agent { φ      :: ActorNet
                   , θ      :: CriticNet  
                   , logStd :: T.IndependentTensor
                   , optim  :: T.Adam
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φ' <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim)

    logStd' <- T.makeIndependent . toFloatGPU $ T.zeros' [1]

    let params = NN.flattenParameters φ' 
              ++ NN.flattenParameters θ' 
              ++ NN.flattenParameters [logStd']
        optim' = T.mkAdam 0 β1 β2 params

    pure $ Agent φ' θ' logStd' optim'

-- | Save an Agent to Disk
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = head $ zipWith T.saveParams [a, c] [pa, pc]
  where
    a  = T.toDependent <$> T.flattenParameters φ
    c  = T.toDependent <$> T.flattenParameters θ
    pa = path ++ "/actor.pt"
    pc = path ++ "/critic.pt"

---- | Load an Actor Net
--loadActor :: String -> Int -> Int -> IO ActorNet
--loadActor fp numObs numAct = T.sample (ActorNetSpec numObs numAct) 
--                           >>= flip T.loadParams fp

---- | Load an Critic Net
--loadCritic :: String -> Int -> Int -> IO CriticNet
--loadCritic fp numObs numAct = T.sample (CriticNetSpec numObs numAct) 
--                            >>= flip T.loadParams fp

-- | Get value and distribution
act :: Agent -> T.Tensor -> (Normal, T.Tensor)
act Agent{..} s = (dist, value)
  where
    value = q θ s
    μ  = π φ s
    σ' = T.exp . T.toDependent $ logStd
    σ  = T.expand σ' True (T.shape μ)
    dist = Normal μ σ

-- | Get value and distribution without grad
act' :: Agent -> T.Tensor -> IO (Normal, T.Tensor)
act' Agent{..} s = do
    value <- T.detach $ q θ s
    μ  <- T.detach $ π φ s
    σ' <- T.detach $ T.exp . T.toDependent $ logStd
    let σ = T.expand σ' True (T.shape μ)
        dist = Normal μ σ
    pure (dist, value)

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Agent -> MemoryLoader T.Tensor -> IO (Agent, T.Tensor)
updateStep agent@Agent{..} MemoryLoader{..} = do

    ((φ', θ', logStd'), optim') <- T.runStep (φ, θ, logStd) optim loss η

    pure (Agent φ' θ' logStd' optim', loss)
  where
    (dist,value) = act agent loaderStates
    entrpy       = T.mean $ entropy dist
    logprobs'    = logProb dist loaderActions
    ratios       = T.exp $ logprobs' - loaderLogPorbs
    surr1        = ratios * loaderAdvantages
    surr2        = loaderAdvantages * T.clamp (1.0 - ε) (1.0 + ε) ratios
    πLoss        = T.mean $ fst . T.minDim (T.Dim 1) T.KeepDim 
                 $ T.cat (T.Dim 1) [surr1, surr2]
    qLoss        = T.mean $ T.pow (2.0 :: Float) (loaderReturns - value)
    loss         = 0.5 * (- πLoss) + qLoss - δ * entrpy
 
-- | Run Policy Update
updatePolicy :: Int -> Int -> Agent -> MemoryLoader [T.Tensor] -> IO Agent
updatePolicy iteration epoch agent loader | loaderLength loader <= 0 = pure agent
                                          | otherwise = do
    (agent', loss) <- updateStep agent batch

    when (verbose && loaderLength loader == 1) do
        putStrLn $ "Iteration " ++ show iteration ++ " Epoch " ++ show epoch 
                               ++ " Loss:\t" ++ show loss
    
    writeLoss iteration "L" (T.asValue loss :: Float)

    updatePolicy iteration epoch agent' loader'
  where
    batch   = head <$> loader
    loader' = tail <$> loader

-- | Evaluation Step
evaluateStep :: Int -> Int -> Agent -> HymURL -> T.Tensor 
             -> ReplayMemory T.Tensor -> T.Tensor 
             -> IO (ReplayMemory T.Tensor, T.Tensor, T.Tensor)
evaluateStep iteration 0 _ _ obs mem total = do
    when verbose do
        putStrLn $ "Iteration " ++ show iteration ++ "\tAverage Reward:\t" 
                ++ show men ++ "\n\t\tTotal Reward:\t" ++ show tot
    pure (mem, obs, total)
  where
    men = T.mean total
    tot = T.sumAll total
evaluateStep iteration step agent envUrl obs mem total = do
    (dist,values') <- act' agent obs

    actions' <- T.clamp (- 1.0) 1.0 <$> sample dist 
    let logprobs' = logProb dist actions'
    (!obs'', !rewards', !dones, !infos) <- stepPool envUrl actions'

    let keys    = head infos
        total'  = T.cat (T.Dim 0) [total, T.reshape [1] . T.mean $ rewards']

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "Environments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t" ++ show de
   
    !obs' <- if T.any dones 
                then flip processGace keys <$> resetPool' envUrl dones
                else pure $ processGace obs'' keys

    let masks' = T.logicalNot dones
        mem'   = memoryPush mem obs' actions' logprobs' rewards' values' masks'

    evaluateStep iteration step' agent envUrl obs' mem' total'
  where
    step' = step - 1

-- | Evaluate Current Policy
evaluatePolicy :: Int -> Agent -> HymURL -> T.Tensor 
               -> IO (ReplayMemory T.Tensor, T.Tensor, T.Tensor)
evaluatePolicy iteration agent envUrl obs = do
    evaluateStep iteration numSteps agent envUrl obs mem tot
  where
    mem = makeMemory
    tot = toTensor ([] :: [Float])

-- | Run Proximal Porlicy Optimization Training
runAlgorithm :: Int -> Agent -> HymURL -> Bool -> T.Tensor -> T.Tensor 
             -> IO Agent
runAlgorithm iteration agent _ True _ reward = do
    putStrLn $ "Finished " ++ show iteration ++ " iterations with total reward: "
            ++ show reward'
    pure agent
  where
    reward' = T.asValue reward :: Float
runAlgorithm iteration agent envUrl _ obs total = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!mem', !obs', !tot') <- evaluatePolicy iteration agent envUrl obs

    let total' = total + T.sumAll tot'
        !loader = dataLoader mem' batchSize γ τ

    agent' <- T.foldLoop agent numEpochs 
                    (\a epoch -> updatePolicy iteration epoch a loader)

    when (iteration `elem` [0,10 .. numIterations]) do
        saveAgent ptPath agent 

    runAlgorithm iteration' agent' envUrl done' obs' total'
  where
    done'      = iteration >= numIterations
    iteration' = iteration + 1
    ptPath     = "./models/" ++ algorithm

-- | Train Proximal Policy Optimization Agent on Environment
train :: Int -> Int -> HymURL -> IO Agent
train obsDim actDim envUrl = do
    remoteLogPath envUrl >>= setupLogging 

    obs' <- toFloatGPU <$> resetPool envUrl
    keys <- infoPool envUrl

    let !obs    = processGace obs' keys
        !reward = toTensor ([] :: [Float])

    !agent <- mkAgent obsDim actDim >>= 
        (\agent' -> runAlgorithm 0 agent' envUrl False obs reward)

    saveAgent ptPath agent 
    pure agent
  where 
      ptPath = "./models/" ++ algorithm

-- | Play Environment with Proximal Policy Optimization Agent
-- play :: Agent -> HymURL -> IO Agent
