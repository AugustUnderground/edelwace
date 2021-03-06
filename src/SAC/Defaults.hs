{-# OPTIONS_GHC -Wall #-}

-- | Soft Actor Critic Algorithm Defaults
module SAC.Defaults where

import Lib
import RPB
import RPB.HER

import qualified Torch as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Algorithm ID
algorithm :: Algorithm
algorithm     = SAC
-- | Print verbose debug output
verbose :: Bool
verbose       = True
-- | Replay Buffer Type
bufferType :: BufferType
bufferType    = HER
-- | How many steps to take in env
numSteps :: Int
numSteps      = 1
-- | How many gradient update steps
numEpochs :: Int
numEpochs     = 40
-- | Total Number of iterations, depends on `bufferType`.
numIterations :: Int
numIterations = 10 ^ (6 :: Int)
-- | Early stop criterion
earlyStop :: T.Tensor
earlyStop     = toTensor (11.0 :: Float)
-- | Reward Lower Bound
minReward :: Float
minReward     = 20.0
-- | Size of the batches during epoch
batchSize :: Int
batchSize     = 128
-- | Random seed for reproducability
rngSeed :: Int
rngSeed       = 666
-- | Maximum time to cut off
maxTime :: Float
maxTime       = 20.0

------------------------------------------------------------------------------
--  ACE Environment Settings
------------------------------------------------------------------------------

-- | ACE Identifier of the Environment
aceId :: String
aceId      = "op2"
-- | PDK/Technology backend of the ACE Environment
aceBackend :: String
aceBackend = "xh035"
-- | ACE Environment variant
aceVariant :: Int
aceVariant = 0

------------------------------------------------------------------------------
--  SAC Algorithm Hyper Parameters
------------------------------------------------------------------------------

-- | Discount Factor
γ :: T.Tensor
γ           = toTensor (0.99 :: Float)
-- | Smoothing Coefficient
τ :: T.Tensor
τ           = toTensor (5.0e-3 :: Float)
-- | Action Noise
εNoise :: T.Tensor
εNoise      = toTensor (1.0e-6 :: Float)
-- | Whether temperature coefficient is fixed or learned (see αInit)
αLearned :: Bool
αLearned    = True
-- | Temperature Coefficient
αInit :: T.Tensor
αInit       = if αLearned
                 then toTensor (0.0 :: Float)
                 else T.log $ toTensor (0.036 :: Float) -- 0.036
-- | Lower Variance Clipping
σMin :: Float
σMin        = -2.0
-- | Upper Variance Clipping
σMax :: Float
σMax        = 20.0
-- | Reward Scaling Factor
rewardScale :: T.Tensor
rewardScale = toTensor (5.0 :: Float)
-- | Reward Scaling Factor
ρ :: T.Tensor
ρ           = toTensor (1.0e-3 :: Float)
-- | Update Step frequency
d :: Int
d           = 2
-- | Priority update factor
εConst :: T.Tensor
εConst      = toTensor (1.0e-5 :: Float)

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Initial weights
wInit :: Float
wInit =  3.0e-3
-- | Learning Rate for Actor / Policy
ηπ :: T.Tensor
ηπ    = toTensor (3.0e-4 :: Float)
-- | Learning Rate for Critic(s)
ηq :: T.Tensor
ηq    = toTensor (3.0e-4 :: Float)
-- | Learning Rate for Alpha
ηα :: T.Tensor
ηα    = toTensor (3.0e-4 :: Float)
-- | Betas
β1   :: Float
β1    = 0.9
-- | Betas
β2   :: Float
β2    = 0.99

------------------------------------------------------------------------------
-- Prioritized Experience Replay Buffer Settings
------------------------------------------------------------------------------

-- | Maximum size of Replay Buffer
bufferSize :: Int
bufferSize = 10 ^ (6 :: Int)
-- | Powerlaw Exponent
αStart :: Float
αStart     = 0.6
-- | Weight Exponent
βStart :: Float
βStart     = 0.4
-- | Weight Exponent Delay
βFrames ::  Int
βFrames    = 10 ^ (5 :: Int)

------------------------------------------------------------------------------
-- Emphasizing Recent Experience Settings
------------------------------------------------------------------------------

-- | Initial η
η0 :: Float
η0 = 0.996
-- | Final η
ηT :: Float
ηT = 1.0
-- | Minimum Sampling Range
cMin :: Int
cMin = 5000

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
