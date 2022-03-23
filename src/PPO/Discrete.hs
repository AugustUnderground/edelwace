{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Proximal Policy Optimization Algorithm
module PPO.Discrete ( algorithm
                    , Agent (..)
                    , mkAgent
                    , saveAgent
                    , loadAgent
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
    a = T.softmax (T.Dim 1) 
      . T.linear pLayer2 . T.tanh 
      . T.linear pLayer1 . T.tanh 
      . T.linear pLayer0 $ o

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
                   , optim  :: T.Adam
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φ' <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim)

    let params = NN.flattenParameters φ' 
              ++ NN.flattenParameters θ' 
        optim' = T.mkAdam 0 β1 β2 params

    pure $ Agent φ' θ' optim'

-- | Save an Agent Checkpoint
saveAgent :: String -> Agent -> IO ()
saveAgent path Agent{..} = do
        T.saveParams φ         (path ++ "/actor.pt")
        T.saveParams θ         (path ++ "/critic.pt")

        saveOptim optim        (path ++ "/optim")

        putStrLn $ "\tSaving Checkpoint at " ++ path ++ " ... "

-- | Load an Agent Checkpoint
loadAgent :: String -> Int -> Int -> Int -> IO Agent
loadAgent path obsDim iter actDim = do
        Agent{..} <- mkAgent obsDim actDim

        fφ      <- T.loadParams φ   (path ++ "/actor.pt")
        fθ      <- T.loadParams θ   (path ++ "/critic.pt")

        fopt     <- loadOptim iter β1 β2 (path ++ "/optim")
       
        pure $ Agent fφ fθ fopt

-- | Get value and distribution
act :: Agent -> T.Tensor -> (D.Categorical, T.Tensor)
act Agent{..} s = (dist, value)
  where
    value = q θ s
    probs = π φ s
    dist  = D.fromProbs probs

-- | Get value and distribution without grad
act' :: Agent -> T.Tensor -> IO (D.Categorical, T.Tensor)
act' Agent{..} s = do
    value <- T.detach $ q θ s
    probs <- T.detach $ π φ s
    let dist = D.fromProbs probs
    pure (dist, value)

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step
updateStep :: Agent -> MemoryLoader T.Tensor -> IO (Agent, T.Tensor)
updateStep agent@Agent{..} MemoryLoader{..} = do

    ((φ', θ'), optim') <- T.runStep (φ, θ) optim loss η

    pure (Agent φ' θ' optim', loss)
  where
    (dist,value) = act agent loaderStates
    entrpy       = D.entropy dist
    logprobs'    = D.logProb dist loaderActions
    ratios       = T.exp $ logprobs' - loaderLogPorbs
    surr1        = ratios * loaderAdvantages
    surr2        = loaderAdvantages * T.clamp (1.0 - ε) (1.0 + ε) ratios
    πLoss        = T.mean $ fst . T.minDim (T.Dim 1) T.KeepDim 
                 $ T.cat (T.Dim 1) [surr1, surr2]
    qLoss        = T.mean $ T.pow (2.0 :: Float) (loaderReturns - value)
    loss         = T.mean $ 0.5 * (- πLoss) + qLoss - δ * entrpy

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
             -> ReplayMemory T.Tensor -> IO (ReplayMemory T.Tensor, T.Tensor)
evaluateStep iteration 0 _ _ states mem = do
    when verbose do
        putStrLn $ "Iteration " ++ show iteration ++ "\tAverage Reward:\t" 
                ++ show men ++ "\n\t\tTotal Reward:\t" ++ show tot
    pure (mem, states)
  where
    men = T.mean . memRewards $ mem
    tot = T.sumAll . memRewards $ mem
evaluateStep iteration step agent envUrl states mem = do
    (dist,values') <- act' agent states

    actions'' <- D.sample dist [1]

    (!states'', !rewards', !dones, !infos) <- stepPool' envUrl actions''

    when (verbose && T.any dones) do
        let de = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "\tEnvironments " ++ " done after " ++ show iteration 
                ++ " iterations, resetting:\n\t\t" ++ show de
   
    let keys = head infos
    !states' <- if T.any dones 
                then flip processGace keys <$> resetPool' envUrl dones
                else pure $ processGace states'' keys

    let masks'    = T.logicalNot dones
        actions'  = T.reshape [-1,1] actions''
        logprobs' = T.reshape [-1,1] $ D.logProb dist actions''
        mem'      = memoryPush mem states' actions' logprobs' rewards' values' masks'

    evaluateStep iteration step' agent envUrl states' mem'
  where
    step' = step - 1

-- | Evaluate Current Policy
evaluatePolicy :: Int -> Agent -> HymURL -> T.Tensor 
               -> IO (ReplayMemory T.Tensor, T.Tensor)
evaluatePolicy iteration agent envUrl states = do
    evaluateStep iteration numSteps agent envUrl states mem
  where
    mem = makeMemory

-- | Run Proximal Porlicy Optimization Training
runAlgorithm :: Int -> Agent -> HymURL -> Bool -> T.Tensor -> IO Agent
runAlgorithm _ agent _ True _ = pure agent
runAlgorithm iteration agent envUrl _ states = do

    when (verbose && iteration `elem` [0,10 .. numIterations]) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!mem', !states') <- evaluatePolicy iteration agent envUrl states

    let !loader = dataLoader mem' batchSize γ τ

    agent' <- T.foldLoop agent numEpochs 
                    (\a epoch -> updatePolicy iteration epoch a loader)

    when (iteration `elem` [0,10 .. numIterations]) do
        saveAgent ptPath agent 

    runAlgorithm iteration' agent' envUrl done' states'
  where
    done'      = iteration >= numIterations
    iteration' = iteration + 1
    ptPath     = "./models/" ++ algorithm

-- | Train Proximal Policy Optimization Agent on Environment
train :: Int -> Int -> HymURL -> IO Agent
train obsDim actDim envUrl = do
    remoteLogPath envUrl >>= setupLogging 

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states = processGace states' keys

    !agent <- mkAgent obsDim actDim >>= 
        (\agent' -> runAlgorithm 0 agent' envUrl False states )

    saveAgent ptPath agent 
    pure agent
  where 
      ptPath = "./models/" ++ algorithm

-- | Play Environment with Proximal Policy Optimization Agent
-- play :: Agent -> HymURL -> IO Agent
