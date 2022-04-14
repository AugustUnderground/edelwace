{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Soft Actor Critic Algorithm Defaults
module SAC ( algorithm
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
           , actRandom
           , act
           , evaluate
           , train
           -- , play
           ) where

import Lib
import SAC.Defaults
import RPB
import qualified RPB.RPB                          as RPB
import qualified RPB.PER                          as PER
import qualified RPB.ERE                          as ERE
import qualified Normal                           as D

import MLFlow                 (TrackingURI)

import Control.Monad
import GHC.Generics
import qualified Torch                            as T
import qualified Torch.Functional.Internal        as T (negative)
import qualified Torch.NN                         as NN
import qualified Torch.Distributions.Distribution as D

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
                         , pLayerσ :: T.Linear 
                         } deriving (Generic, Show, T.Parameterized)

-- | Critic Network Architecture
data CriticNet = CriticNet { qLayer0 :: T.Linear
                           , qLayer1 :: T.Linear
                           , qLayer2 :: T.Linear 
                           } deriving (Generic, Show, T.Parameterized)

-- | Actor Network Weight initialization
instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec{..} = ActorNet <$> ( T.sample (T.LinearSpec pObsDim 256) 
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 256     256)
                                             >>= weightInitUniform' )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
                                             >>= weightInitUniform (-wInit) wInit )
                                       <*> ( T.sample (T.LinearSpec 256 pActDim)
                                             >>= weightInitUniform (-wInit) wInit )

-- | Critic Network Weight initialization
instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec{..} = CriticNet <$> ( T.sample (T.LinearSpec dim 256) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 256 256) 
                                               >>= weightInitUniform' )
                                         <*> ( T.sample (T.LinearSpec 256 1) 
                                               >>= weightInitUniform' )
        where dim = qObsDim + qActDim

-- | Actor Network Forward Pass
π :: ActorNet -> T.Tensor -> (T.Tensor, T.Tensor)
π ActorNet{..} s = (μ, σ)
  where
    x = T.relu . T.linear pLayer1 
      . T.relu . T.linear pLayer0 $ s
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
data Agent = Agent { φ       :: ActorNet            -- ^ Actor policy φ
                   , θ1      :: CriticNet           -- ^ Online Critic θ1
                   , θ2      :: CriticNet           -- ^ Online Critic θ2
                   , θ1'     :: CriticNet           -- ^ Target Critic θ'1
                   , θ2'     :: CriticNet           -- ^ Target Critic θ'2
                   , φOptim  :: T.Adam              -- ^ Policy Optimizer
                   , θ1Optim :: T.Adam              -- ^ Critic 1 Optimizer
                   , θ2Optim :: T.Adam              -- ^ Critic 2 Optimizer
                   , h'      :: Float               -- ^ Target Entropy
                   , αLog    :: T.IndependentTensor -- ^ Temperature Coefficient
                   , αOptim  :: T.Adam              -- ^ Alpha Optimizer
                   } deriving (Generic, Show)

-- | Agent constructor
mkAgent :: Int -> Int -> IO Agent
mkAgent obsDim actDim = do
    φOnline   <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    θ1Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Online  <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ1Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    θ2Target' <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)

    αlog <- T.makeIndependent αInit

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

-- | Save an Agent and return the agent
saveAgent' :: String -> Agent -> IO Agent
saveAgent' p a = saveAgent p a >> pure a

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

-- transferAgent :: Agent -> Agent -> IO Agent
-- transferAgent source@Agent{..} target = pure source

-- | Perform a completely random action for a given state
actRandom :: Agent -> T.Tensor -> IO T.Tensor
actRandom Agent{..} s = (\a' -> a' * 2.0 - 1.0) . toFloatGPU <$> T.randLikeIO' μ
  where
    (μ,_) = π φ s

-- | Get an Action (no grad)
act :: Agent -> T.Tensor -> IO T.Tensor
act Agent{..} s = do
    ε <- toFloatGPU <$> T.randnLikeIO μ                    -- different ε's
    -- ε <- toFloatGPU <$> T.randnIO' [head . T.shape $ μ, 1] -- same ε per sample
    -- ε <- toFloatGPU <$> T.randnIO' [(!!1) . T.shape $ μ]   -- same ε per action
    T.detach . T.tanh $ (μ + σ * ε)
  where
    (μ,σ') = π φ s
    σ      = T.exp σ'

-- | Get an action and log probs (grad)
evaluate :: Agent -> T.Tensor -> T.Tensor -> IO (T.Tensor, T.Tensor)
evaluate Agent{..} s εN = do
    ε <- toFloatGPU <$> T.randnLikeIO μ                    -- different ε's
    -- ε <- toFloatGPU <$> T.randnIO' [head . T.shape $ μ, 1] -- same ε per sample
    -- ε <- toFloatGPU <$> T.randnIO' [(!!1) . T.shape $ μ]      -- same ε per action
    let a' = μ + σ * ε
        a  = T.tanh a'
        l1 = D.logProb n a'
        l2 = T.log . T.abs $ 1.0 - T.pow (2.0 :: Float) a + εN
        -- p  = T.meanDim (T.Dim 1) T.KeepDim T.Float $ l1 - l2
        p  = T.sumDim (T.Dim 1) T.KeepDim T.Float $ l1 - l2
    pure (a,p)
  where
    (μ,σ') = π φ s
    σ      = T.exp σ'
    n      = D.Normal μ σ

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Policy Update Step (PER)
updateStepPER :: Int -> Int -> Agent -> Tracker -> RPB.Buffer T.Tensor 
              -> T.Tensor -> T.Tensor -> IO (Agent, T.Tensor)
updateStepPER _ 0 agent _ _ _ prios = pure (agent, prios)
updateStepPER iteration epoch agent@Agent{..} tracker memories@RPB.Buffer{..} weights _ = do
    let αLog' = T.toDependent αLog
        α     = T.exp αLog'
    α' <- if iteration == 0 then pure $ toTensor (0.0 :: Float)
                            else T.detach α

    (a_t1, logπ_t1) <- evaluate agent s_t1 εNoise

    let q_t1' = q' θ1' θ2' s_t1 a_t1
    q_t1 <- T.detach (r + (γ * d' * (q_t1' - α' * logπ_t1))) >>= T.clone

    let q1_t0 = q θ1 s_t0 a_t0
        q2_t0 = q θ2 s_t0 a_t0

    let δ1 = w * T.mseLoss q1_t0 q_t1
        δ2 = w * T.mseLoss q2_t0 q_t1

    jQ1 <- T.clone . T.mean $ 0.5 * δ1
    jQ2 <- T.clone . T.mean $ 0.5 * δ2

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηq
    (θ2Online', θ2Optim') <- T.runStep θ2 θ2Optim jQ2 ηq
    
    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "\tQ1 Loss:\t" ++ show jQ1
        putStrLn $ "\tQ2 Loss:\t" ++ show jQ2

    _ <- trackLoss tracker iteration "Q1" (T.asValue jQ1 :: Float)
    _ <- trackLoss tracker iteration "Q2" (T.asValue jQ2 :: Float)

    (a_t0', logπ_t0') <- evaluate agent s_t0 εNoise

    let updateAlpha :: IO(T.IndependentTensor, T.Adam)
        updateAlpha = do
            logπ_t0 <- T.clone logπ_t0' >>= T.detach
            -- let jα = T.mean (- α * logπ_t0 - α * h)
            let jα = T.negative . T.mean $ αLog' * (logπ_t0 + h)
            when (verbose && iteration `mod` 10 == 0) do
                putStrLn $ "\tα  Loss:\t" ++ show jα
            _ <- trackLoss tracker iteration "alpha" (T.asValue jα :: Float)
            T.runStep αLog αOptim jα ηα
        updateActor :: IO(ActorNet, T.Adam)
        updateActor = do
            q_t0' <- T.detach $ q' θ1 θ2 s_t0 a_t0'
            let jπ = T.mean $ (α' * logπ_t0') - q_t0'
            when (verbose && iteration `mod` 10 == 0) do
                putStrLn $ "\tπ  Loss:\t" ++ show jπ
            _ <- trackLoss tracker iteration  "policy" (T.asValue jπ :: Float)
            T.runStep φ φOptim jπ ηπ
        syncCritic :: IO (CriticNet, CriticNet)
        syncCritic = do
            θ1Target' <- softSync τ θ1' θ1 
            θ2Target' <- softSync τ θ2' θ2 
            pure (θ1Target', θ2Target')

    (αlog', αOptim') <- if iteration `mod` d == 0 && αLearned
                           then updateAlpha
                           else pure (αLog, αOptim)

    (φOnline', φOptim') <- if iteration `mod` d == 0
                              then updateActor
                              else pure (φ, φOptim)

    (θ1Target', θ2Target') <- if iteration `mod` d == 0
                                 then syncCritic
                                 else pure (θ1', θ2')

    prios' <- T.detach $ T.abs (0.5 * (δ1 + δ2) + εConst)

    let agent' = Agent φOnline' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'  θ1Optim'  θ2Optim'  h' αlog' αOptim'

    updateStepPER iteration epoch' agent' tracker memories weights prios'
  where
    epoch' = epoch - 1
    s_t0   = states
    a_t0   = actions
    s_t1   = states'
    d'     = toFloatGPU (1.0 - dones)
    w      = T.reshape [-1,1] weights
    h      = toTensor h'
    r      = rewards * rewardScale
    -- r      = (rewardScale *) . fst . T.minDim (T.Dim 1) T.KeepDim 
    --        $ T.cat (T.Dim 1) [RPB.rewards, fullLike' RPB.rewards minReward]
    -- r    = scaleRewards rewards ρ

-- | Perform Policy Update Steps (PER)
updatePolicyPER :: Int -> Agent -> Tracker -> PER.Buffer T.Tensor -> Int 
                -> IO (PER.Buffer T.Tensor, Agent)
updatePolicyPER iteration agent tracker buffer epochs = do
    (memories, indices, weights) <- PER.sampleIO buffer iteration batchSize
    let prios = PER.priorities buffer
    (agent', prios') <- updateStepPER iteration epochs agent tracker memories 
                                      weights prios
    let buffer' = PER.update buffer indices prios'
    pure (buffer', agent')

-- | Policy Update Step (RPB)
updateStepRPB :: Int -> Int -> Agent -> Tracker -> RPB.Buffer T.Tensor 
              -> IO Agent
updateStepRPB _ 0 agent _ _ = pure agent
updateStepRPB iteration epoch agent@Agent{..} tracker memories@RPB.Buffer{..} = do
    let αLog' = T.toDependent αLog
        α     = T.exp αLog'
    α' <- if iteration == 0 then pure $ toTensor (0.0 :: Float)
                            else T.detach α

    (a_t1, logπ_t1) <- evaluate agent s_t1 εNoise

    let q_t1' = q' θ1' θ2' s_t1 a_t1
    q_t1 <- T.detach (r + (γ * d' * (q_t1' - α' * logπ_t1))) >>= T.clone

    let q1_t0 = q θ1 s_t0 a_t0
        q2_t0 = q θ2 s_t0 a_t0

    let δ1 = T.mseLoss q1_t0 q_t1
        δ2 = T.mseLoss q2_t0 q_t1

    jQ1 <- T.clone . T.mean $ 0.5 * δ1
    jQ2 <- T.clone . T.mean $ 0.5 * δ2

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηq
    (θ2Online', θ2Optim') <- T.runStep θ2 θ2Optim jQ2 ηq
    
    when (verbose && (bufferType == RPB || epoch `mod` 4 == 0)) do
        putStrLn $ "\tQ1 Loss:\t" ++ show jQ1
        putStrLn $ "\tQ2 Loss:\t" ++ show jQ2

    _ <- trackLoss tracker iteration "Q1" (T.asValue jQ1 :: Float)
    _ <- trackLoss tracker iteration "Q2" (T.asValue jQ2 :: Float)

    (a_t0', logπ_t0') <- evaluate agent s_t0 εNoise

    let updateAlpha :: IO(T.IndependentTensor, T.Adam)
        updateAlpha = do
            logπ_t0 <- T.clone logπ_t0' >>= T.detach
            let jα = T.negative . T.mean $ αLog' * (logπ_t0 + h)
            when (verbose && (bufferType == RPB || epoch `mod` 4 == 0)) do
                putStrLn $ "\tα  Loss:\t" ++ show jα
            _ <- trackLoss tracker iteration "alpha" (T.asValue jα :: Float)
            T.runStep αLog αOptim jα ηα
        updateActor :: IO(ActorNet, T.Adam)
        updateActor = do
            q_t0' <- T.detach $ q' θ1 θ2 s_t0 a_t0'
            let jπ = T.mean $ (α' * logπ_t0') - q_t0'
            when (verbose && (bufferType == RPB || epoch `mod` 4 == 0)) do
                putStrLn $ "\tπ  Loss:\t" ++ show jπ
            _ <- trackLoss tracker iteration "policy" (T.asValue jπ :: Float)
            T.runStep φ φOptim jπ ηπ
        syncCritic :: IO (CriticNet, CriticNet)
        syncCritic = do
            θ1Target' <- softSync τ θ1' θ1 
            θ2Target' <- softSync τ θ2' θ2 
            pure (θ1Target', θ2Target')

    (αlog', αOptim') <- if iteration `mod` d == 0 && αLearned
                           then updateAlpha
                           else pure (αLog, αOptim)

    (φOnline', φOptim') <- if iteration `mod` d == 0
                              then updateActor
                              else pure (φ, φOptim)

    (θ1Target', θ2Target') <- if iteration `mod` d == 0
                                 then syncCritic
                                 else pure (θ1', θ2')

    let agent' = Agent φOnline' θ1Online' θ2Online' θ1Target' θ2Target' 
                       φOptim'  θ1Optim'  θ2Optim'  h' αlog' αOptim'

    updateStepRPB iteration epoch' agent' tracker memories
  where
    epoch' = epoch - 1
    s_t0   = states
    a_t0   = actions
    s_t1   = states'
    d'     = toFloatGPU (1.0 - dones)
    h      = toTensor h'
    r      = rewards * rewardScale
    -- r      = (rewardScale *) . fst . T.minDim (T.Dim 1) T.KeepDim 
    --        $ T.cat (T.Dim 1) [RPB.rewards, fullLike' RPB.rewards minReward]
    -- r      = scaleRewards rewards ρ

-- | Perform Policy Update Steps (RPB)
updatePolicyRPB :: Int -> Agent -> Tracker -> RPB.Buffer T.Tensor -> Int 
                -> IO Agent
updatePolicyRPB iteration agent tracker buffer epochs =
    RPB.sampleIO batchSize buffer >>= 
        updateStepRPB iteration epochs agent tracker

-- | Policy Update Step (ERE)
updateStepERE :: Int -> Int -> Int -> Agent -> Tracker -> RPB.Buffer T.Tensor 
              -> IO Agent
updateStepERE _ 0 _ agent _ _ = pure agent
updateStepERE iteration epoch _ agent@Agent{..} tracker RPB.Buffer{..} = do
    let αLog' = T.toDependent αLog
        α     = T.exp αLog'
    α' <- if iteration == 0 then pure $ toTensor (0.0 :: Float)
                            else T.detach α

    (a_t1, logπ_t1) <- evaluate agent s_t1 εNoise

    let q_t1' = q' θ1' θ2' s_t1 a_t1
    q_t1 <- T.detach (r + (γ * d' * (q_t1' - α' * logπ_t1))) >>= T.clone

    let q1_t0 = q θ1 s_t0 a_t0
        q2_t0 = q θ2 s_t0 a_t0

    let δ1 = T.mseLoss q1_t0 q_t1
        δ2 = T.mseLoss q2_t0 q_t1

    jQ1 <- T.clone . T.mean $ 0.5 * δ1
    jQ2 <- T.clone . T.mean $ 0.5 * δ2

    (θ1Online', θ1Optim') <- T.runStep θ1 θ1Optim jQ1 ηq
    (θ2Online', θ2Optim') <- T.runStep θ2 θ2Optim jQ2 ηq
    
    when (verbose && epoch `mod` 4 == 0) do
        putStrLn $ "\tEpoch " ++ show epoch
        putStrLn $ "\t\tQ1 Loss:\t" ++ show jQ1
        putStrLn $ "\t\tQ2 Loss:\t" ++ show jQ2

    _ <- trackLoss tracker (iteration + epoch) "Q1" (T.asValue jQ1 :: Float)
    _ <- trackLoss tracker (iteration + epoch) "Q2" (T.asValue jQ2 :: Float)

    (a_t0', logπ_t0') <- evaluate agent s_t0 εNoise

    logπ_t0 <- T.clone logπ_t0' >>= T.detach
    let jα = T.negative . T.mean $ αLog' * (logπ_t0 + h)

    when (verbose && epoch `mod` 4 == 0) do
        putStrLn $ "\t\tα  Loss:\t" ++ show jα
    _ <- trackLoss tracker (iteration + epoch) "alpha" (T.asValue jα :: Float)

    (αlog', αOptim') <- T.runStep αLog αOptim jα ηα

    q_t0' <- T.detach $ q' θ1 θ2 s_t0 a_t0'
    let jπ = T.mean $ (α' * logπ_t0') - q_t0'

    when (verbose && epoch `mod` 4 == 0) do
        putStrLn $ "\t\tπ  Loss:\t" ++ show jπ
    _ <- trackLoss tracker (iteration + epoch) "policy" (T.asValue jπ :: Float)

    (φOnline', φOptim') <- T.runStep φ φOptim jπ ηπ

    θ1Target' <- softSync τ θ1' θ1 
    θ2Target' <- softSync τ θ2' θ2 

    pure $ Agent φOnline' θ1Online' θ2Online' θ1Target' θ2Target' 
                 φOptim'  θ1Optim'  θ2Optim'  h' αlog' αOptim'
  where
    s_t0   = states
    a_t0   = actions
    s_t1   = states'
    d'     = toFloatGPU (1.0 - dones)
    h      = toTensor h'
    r      = rewards * rewardScale
    -- r      = (rewardScale *) . fst . T.minDim (T.Dim 1) T.KeepDim 
    --        $ T.cat (T.Dim 1) [RPB.rewards, fullLike' RPB.rewards minReward]
    -- r      = scaleRewards rewards ρ

-- | Sample for ERE Buffer and perform one update step
updatePolicyERE :: Int -> Int -> Int -> Tracker -> RPB.Buffer T.Tensor 
                -> Float -> Agent -> IO Agent
updatePolicyERE iteration epoch epochs tracker buffer ηt agent 
                | epoch > epochs = pure agent
                | otherwise      =
        ERE.sample buffer bufferSize batchSize epochs epoch cMin ηt >>= 
            updateStepERE iteration epoch epochs agent tracker >>=
                updatePolicyERE iteration epoch' epochs tracker buffer ηt
  where
    epoch' = epoch + 1

-- | Buffer independent exploration step in the environment
evaluateStep :: Int -> Int -> Agent -> HymURL -> Tracker -> T.Tensor 
             -> IO (T.Tensor, T.Tensor, T.Tensor, T.Tensor)
evaluateStep iteration _ agent envUrl tracker states = do
    actions <- act agent states
    (!states'', !rewards, !dones, !infos) <- stepPool envUrl actions
    
    _ <- trackReward tracker iteration rewards
    when (even iteration) do
        _ <- trackEnvState tracker envUrl iteration
        pure ()
 
    when (verbose && T.any dones) do
        let de   = T.squeezeAll . T.nonzero . T.squeezeAll $ dones
        putStrLn $ "\tEnvironments finished episode after " ++ show iteration 
                ++ " iterations, resetting:\n\t\t" ++ show de
   
    let keys = head infos
    !states' <- if T.any dones 
                   then flip processGace keys <$> resetPool' envUrl dones
                   else pure $ processGace states'' keys

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "\tAverage Reward:\t" ++ show (T.mean rewards)

    pure (actions, rewards, states', dones)

-- | Take steps in the Environment, evaluating the current policy (PER)
evaluatePolicyPER :: Int -> Int -> Agent -> HymURL -> Tracker -> PER.Buffer T.Tensor
                  -> T.Tensor -> IO (PER.Buffer T.Tensor, T.Tensor)
evaluatePolicyPER _ 0 _ _ _ buffer states = pure (buffer, states)
evaluatePolicyPER iteration step agent envUrl tracker buffer states = do

    (actions, rewards, states', dones) <- 
            evaluateStep iteration step agent envUrl tracker states
    let buffer' = PER.push buffer states actions rewards states' dones

    evaluatePolicyPER iteration step' agent envUrl tracker buffer' states'
  where
    step' = step - 1

-- | Take steps in the Environment, evaluating the current policy (RPB)
evaluatePolicyRPB :: Int -> Int -> Agent -> HymURL -> Tracker -> RPB.Buffer T.Tensor
                  -> T.Tensor -> IO (RPB.Buffer T.Tensor, T.Tensor)
evaluatePolicyRPB _ 0 _ _ _ buffer states = pure (buffer, states)
evaluatePolicyRPB iteration step agent envUrl tracker buffer states = do
    (actions, rewards, states', dones) <- 
            evaluateStep iteration step agent envUrl tracker states
    let buffer' = RPB.push bufferSize buffer states actions rewards states' dones

    evaluatePolicyRPB iteration step' agent envUrl tracker buffer' states'
  where
    step' = step - 1

-- | Run Soft Actor Critic Training (PER)
runAlgorithmPER :: Int -> Agent -> HymURL -> Tracker -> Bool -> PER.Buffer T.Tensor
                -> T.Tensor -> IO Agent
runAlgorithmPER _ agent _ _ True _ _ = pure agent
runAlgorithmPER iteration agent envUrl tracker _ buffer states = do

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!memories', !states') <- evaluatePolicyPER iteration numSteps agent envUrl 
                                                tracker buffer states

    let buffer'' = PER.push' buffer memories'
        bufLen   = RPB.size . PER.memories $ buffer''

    (!buffer', !agent') <- if bufLen < batchSize 
                              then pure (buffer'', agent)
                              else updatePolicyPER iteration agent tracker
                                                   buffer'' numEpochs
    
    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . RPB.rewards . PER.memories $ memories'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithmPER iteration' agent' envUrl tracker done' buffer' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ show algorithm

-- | Run Soft Actor Critic Training (RPB)
runAlgorithmRPB :: Int -> Agent -> HymURL -> Tracker -> Bool 
                -> RPB.Buffer T.Tensor -> T.Tensor -> IO Agent
runAlgorithmRPB _ agent _ _ True _ _ = pure agent
runAlgorithmRPB iteration agent envUrl tracker _ buffer states = do

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!memories', !states') <- evaluatePolicyRPB iteration numSteps agent envUrl 
                                                tracker buffer states

    let buffer' = RPB.push' bufferSize buffer memories'
        bufLen  = RPB.size buffer'

    !agent' <- if bufLen < batchSize 
                  then pure agent
                  else updatePolicyRPB iteration agent tracker buffer' numEpochs
    
    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . RPB.rewards $ memories'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithmRPB iteration' agent' envUrl tracker done' buffer' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ show algorithm

runAlgorithmERE :: Int -> Int -> Int -> Agent -> HymURL -> Tracker -> Bool 
                -> RPB.Buffer T.Tensor -> T.Tensor -> IO Agent
runAlgorithmERE _ _ _ agent _ _ True _ _ = pure agent
runAlgorithmERE iteration epochs numEnvs agent envUrl tracker _ buffer states = do

    when (verbose && iteration `mod` 10 == 0) do
        putStrLn $ "Iteration " ++ show iteration ++ " / " ++ show numIterations
    
    (!memories', !states') <- evaluatePolicyRPB iteration numSteps agent envUrl 
                                                tracker buffer states

    let buffer' = RPB.push' bufferSize buffer memories'
        dones   = T.any . RPB.dones $ RPB.pop numEnvs memories'
        epochs' = if dones then 0 else epochs + 1
        ηt      = ERE.anneal η0 ηT numIterations iteration

    when (dones && epochs > 1) do
        putStrLn $ "\tRun update for " ++ show epochs ++ " Epochs."

    !agent' <- if dones && epochs > 1
                  then updatePolicyERE iteration 0 epochs tracker buffer ηt agent
                  else pure agent
    
    when (iteration `mod` 10 == 0) do
        saveAgent ptPath agent 

    let meanReward = T.mean . RPB.rewards $ memories'
        stop       = T.asValue (T.ge meanReward earlyStop) :: Bool
        done'      = (iteration >= numIterations) || stop

    runAlgorithmERE iteration' epochs' numEnvs agent' envUrl tracker done' buffer' states'
  where
    iteration' = iteration + 1
    ptPath     = "./models/" ++ show algorithm

-- | Handle training for different replay buffer types
train' :: HymURL -> Tracker -> BufferType -> T.Tensor -> Agent -> IO Agent
train' envUrl tracker PER states agent = do
    let !buffer = PER.mkBuffer bufferSize αStart βStart βFrames
    runAlgorithmPER 0 agent envUrl tracker False buffer states
train' envUrl tracker RPB states agent = 
    runAlgorithmRPB 0 agent envUrl tracker False RPB.mkBuffer states 
train' envUrl tracker ERE states agent = do
    numEnvs <- numEnvsPool envUrl
    runAlgorithmERE 0 0 numEnvs agent envUrl tracker False RPB.mkBuffer states 
-- train' envUrl tracker PERERE states agent = error "Not Implemented"
train' _ _ _ _ _ = undefined

-- | Train Soft Actor Critic Agent on Environment
train :: Int -> Int -> HymURL -> TrackingURI -> IO Agent
train obsDim actDim envUrl trackingUri = do
    numEnvs <- numEnvsPool envUrl
    tracker <- mkTracker trackingUri (show algorithm) >>= newRuns' numEnvs

    states' <- toFloatGPU <$> resetPool envUrl
    keys    <- infoPool envUrl

    let !states = processGace states' keys

    !agent <- mkAgent obsDim actDim >>= train' envUrl tracker bufferType states
    createModelArchiveDir (show algorithm) >>= (`saveAgent` agent)

    endRuns' tracker

    pure agent

-- | Play Environment with Soft Actor Critic Agent
-- play :: Agent -> HymURL -> IO Agent
