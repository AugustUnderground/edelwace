{-# OPTIONS_GHC -Wall #-}

-- | Proximal Policy Optimization Algorithm Defaults
module PPO.Defaults where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Algorithm ID
algorithm :: String
algorithm     = "ppo"
-- | Print verbose debug output
verbose :: Bool
verbose       = True
-- | Number of episodes to play
numEpisodes :: Int
numEpisodes   = 666
-- | How many steps to take in env
numSteps :: Int
numSteps      = 64
-- | How many gradient update steps
numEpochs :: Int
numEpochs     = 16
-- | Number of iterations
numIterations :: Int
numIterations = round (1.0e6 :: Float)
-- | Early stop criterion
earlyStop :: Float
earlyStop     = -500.0
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
--  PPO Algorithm Hyper Parameters
------------------------------------------------------------------------------

-- | Discrete or Continuous action space
actionSpace :: ActionSpace
actionSpace = Discrete
-- | Factor for clipping
ε :: Float
ε        = 0.2
-- | Factor in loss function
δ :: T.Tensor
δ        = toTensor (0.001 :: Float)
-- | Discount Factor
γ :: T.Tensor
γ        = toTensor (0.99 :: Float)
-- | Avantage Factor
τ :: T.Tensor
τ        = toTensor (0.95 :: Float)

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Initial weights
wInit :: Float
wInit = 3.0e-3
-- | Learning Rate
η :: T.Tensor
η  = toTensor (2.5e-4 :: Float)
-- | Betas
β1 :: Float
β1 = 0.9
-- | Betas
β2 :: Float
β2 = 0.999
