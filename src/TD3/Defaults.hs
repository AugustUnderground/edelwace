{-# OPTIONS_GHC -Wall #-}

-- | Twin Delayed Deep Deterministic Policy Gradient Algorithm Defaults
module TD3.Defaults where

import Lib
import RPB
import RPB.HER

import qualified Torch   as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Algorithm ID
algorithm :: Algorithm
algorithm     = TD3
-- | Print verbose debug output
verbose :: Bool
verbose       = True
-- | Number of episodes to play
numEpisodes :: Int
numEpisodes   = 666
-- | Horizon T
numIterations :: Int
numIterations = 500 -- 10 ^ (8 :: Int)
-- | Number of Steps to take with policy
numSteps :: Int
numSteps      = 50
-- | Number of epochs to train
numEpochs :: Int
numEpochs     = 40
-- | Early stop criterion
earlyStop :: T.Tensor
earlyStop     = toTensor (11.0 :: Float)
-- | Mini batch of N transistions
batchSize :: Int
batchSize     = 128
-- | Random seed for reproducability
rngSeed :: Int
rngSeed       = 666

------------------------------------------------------------------------------
--  ACE Environment Settings
------------------------------------------------------------------------------
           
-- | ACE Identifier of the Environment
aceId :: String
aceId      = "op2"
-- |  PDK/Technology backend of the ACE Environment
aceBackend :: String
aceBackend = "xh035"
-- | ACE Environment variant
aceVariant :: Int
aceVariant = 0
-- | Action space lower bound
actionLow :: Float
actionLow  = - 1.0
-- | Action space upper bound
actionHigh :: Float
actionHigh = 1.0

------------------------------------------------------------------------------
--  TD3 Algorithm Hyper Parameters
------------------------------------------------------------------------------

-- | Policy and Target Update Delay
d :: Int
d           = 2
-- | Noise clipping
c :: Float
c           = 0.5
-- | Discount Factor
γ :: T.Tensor
γ           = toTensor (0.98 :: Float)
-- | Avantage Factor
τ :: T.Tensor
τ           = toTensor (0.95 :: Float) -- (1.0e-2 :: Float)
-- | Decay Period
decayPeriod :: Int
decayPeriod = 10 ^ (5 :: Int)
-- | Noise Clipping Minimum
σMin :: Float
σMin        = 1.0
-- | Noise Clipping Maximuxm
σMax :: Float
σMax        = 1.0
-- | Evaluation Noise standard deviation (σ~)
σEval :: T.Tensor
σEval       = toTensor ([0.2] :: [Float])
-- | Action Noise standard deviation
σAct :: T.Tensor
σAct        = toTensor ([0.1] :: [Float])

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Initial weights
wInit :: Float
wInit = 3.0e-3
-- | Learning Rate
ηθ :: T.Tensor
ηθ    = toTensor (1.0e-3 :: Float)
-- | Learning Rate
ηφ :: T.Tensor
ηφ    = toTensor (1.0e-3 :: Float)
-- | Betas
β1   :: Float
β1    = 0.9
-- | Betas
β2   :: Float
β2    = 0.99

------------------------------------------------------------------------------
--  Memory / Replay Buffer Settings
------------------------------------------------------------------------------

-- | Replay Buffer Type
bufferType :: BufferType
bufferType    = HER
-- | Replay Buffer Size
bufferSize :: Int
bufferSize    = 10 ^ (6 :: Int)
-- | Initial sample collecting period
warmupPeriode :: Int
warmupPeriode = 50
-- | Range for clipping scaled states
stateClip :: Float
stateClip = 5.0

------------------------------------------------------------------------------
-- Hindsight Experience Replay Settings
------------------------------------------------------------------------------

-- | Target Sampling Strategy
strategy :: Strategy
strategy = Future
-- | Number of Additional Targets to sample
k :: Int 
k        = 4
-- | Error Tolerance for Target / Reward Calculation
relTol :: T.Tensor
relTol   = toTensor (1.0e-4 :: Float)
