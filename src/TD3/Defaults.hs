{-# OPTIONS_GHC -Wall #-}

module TD3.Defaults where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Algorithm ID
algorithm :: String
algorithm     = "td3"
-- | Print verbose debug output
verbose :: Bool
verbose       = True
-- | Number of episodes to play
numEpisodes :: Int
numEpisodes   = 666
-- | Horizon T
numIterations :: Int
numIterations = 150
-- | Number of Steps to take with policy
numSteps :: Int
numSteps = 1
-- | Number of epochs to train
numEpochs :: Int
numEpochs = 1
-- | Early stop criterion
earlyStop :: Float
earlyStop     = -500.0
-- | Mini batch of N transistions
batchSize :: Int
batchSize     = 100
-- | Random seed for reproducability
rngSeed :: Int
rngSeed       = 666

------------------------------------------------------------------------------
--  ACE Environment Settings
------------------------------------------------------------------------------
           
-- | ACE Identifier of the Environment
aceId :: String
aceId       = "op2"
-- |  PDK/Technology backend of the ACE Environment
aceBackend :: String
aceBackend = "xh035"
-- | ACE Environment variant
aceVariant :: Int
aceVariant  = 0
-- | Number of Environments
numEnvs :: Int
numEnvs    = 20

------------------------------------------------------------------------------
--  TD3 Algorithm Hyper Parameters
------------------------------------------------------------------------------

-- | Policy and Target Critic Update Delay
d :: Int
d           = 2
-- | Noise clipping
c :: Float
c           = 0.5
-- | Discount Factor
γ :: T.Tensor
γ           = toTensor (0.99 :: Float)
-- | Avantage Factor
τ :: T.Tensor
τ           = toTensor (1.0e-2 :: Float)
-- | Sampling Noise as Tensor
σ :: T.Tensor
σ           = toTensor (0.2 :: Float)
-- | Decay Period
decayPeriod :: Int
decayPeriod = round (1.0e5 :: Float)
-- | Noise Clipping Minimum
σMin :: Float
σMin        = 1
-- | Noise Clipping Maximuxm
σMax :: Float
σMax        = 1

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Initial weights
wInit :: Float
wInit = 3.0e-3
-- | Learning Rate
ηθ :: T.Tensor
ηθ     = toTensor (1.0e-4 :: Float)
-- | Learning Rate
ηφ :: T.Tensor
ηφ     = toTensor (1.0e-4 :: Float)
-- | Optimizer Betas
β1   :: Float
β1    = 0.9
β2   :: Float
β2    = 0.99

------------------------------------------------------------------------------
--  Memory / Replay Buffer Settings
------------------------------------------------------------------------------

-- | Replay Buffer Size
bufferSize :: Int
bufferSize    = round (1.0e7 :: Float)
-- | Initial sample collecting period
warmupPeriode :: Int
warmupPeriode = 50
