{-# OPTIONS_GHC -Wall #-}

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

