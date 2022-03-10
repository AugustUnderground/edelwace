{-# OPTIONS_GHC -Wall #-}

module SAC.Defaults where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
--  General Default Settings
------------------------------------------------------------------------------

-- | Algorithm ID
algorithm :: String
algorithm     = "sac"
-- | Print verbose debug output
verbose :: Bool
verbose       = True
-- | Number of episodes to play
numEpisodes :: Int
numEpisodes   = 666
-- | How many steps to take in env
numSteps :: Int
numSteps      = 1
-- | How many gradient update steps
numEpochs :: Int
numEpochs     = 1
-- | Number of iterations
numIterations :: Int
numIterations = 150
-- | Early stop criterion
earlyStop :: Float
earlyStop     = -500.0
-- | Size of the batches during epoch
batchSize :: Int
batchSize     = 256
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
-- | Number of Environments
numEnvs :: Int
numEnvs    = 20

------------------------------------------------------------------------------
--  SAC Algorithm Parameter Settings
------------------------------------------------------------------------------

-- | Discount Factor
γ :: Float
γ           = 0.99
-- | Smoothing Coefficient
τSoft :: Float
τSoft       = 1.0e-2
-- | Action Noise
εNoise :: Float
εNoise      = 1.0e-6
-- | Temperature Parameter
αConst :: Float
αConst      = 0.2   -- 3.0e-4
-- | Lower Variance Clipping
σMin :: Float
σMin        = -2.0
-- | Upper Variance Clipping
σMax :: Float
σMax        = 20.0
-- | Reward Scaling Factor
rewardScale :: Float
rewardScale = 5.0
-- | Reward Scaling Factor
ρ :: Float
ρ           = 1.0e-3
-- | Update Step frequency
d :: Int
d = 1
-- | Priority update factor
εConst :: Float
εConst      = 1.0e-5

------------------------------------------------------------------------------
-- Neural Network Parameter Settings
------------------------------------------------------------------------------

-- | Initial weights
wInit :: Float
wInit = 3.0e-3
-- | Learning Rate for Actor / Policy
ηπ :: T.Tensor
ηπ    = toTensor (1.0e-4 :: Float)
-- | Learning Rate for Critic(s)
ηq :: T.Tensor
ηq    = toTensor (3.0e-4 :: Float)
-- | Learning Rate for Alpha
ηα :: T.Tensor
ηα    = toTensor (3.0e-4 :: Float)
-- | Optimizer Betas
β1   :: Float
β1    = 0.9
β2   :: Float
β2    = 0.99

------------------------------------------------------------------------------
-- Prioritized Experience Replay Buffer Settings
------------------------------------------------------------------------------

-- | Maximum size of Replay Buffer
bufferSize :: Int
bufferSize = round (1.0e6 :: Float)
-- | Powerlaw Exponent
αStart :: Float
αStart     = 0.6
-- | Weight Exponent
βStart :: Float
βStart     = 0.4
-- | Weight Exponent Delay
βFrames ::  Int
βFrames    = round (1.0e5 :: Float)