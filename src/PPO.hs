{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module PPO ( algorithm
           --, Agent
           --, makeAgent
           --, saveAgent
           --, π
           --, q
           --, q'
           --, addNoise
           --, train
           -- , play
           ) where

import Lib
import RPB
import PPO.Defaults

import Control.Monad
import GHC.Generics
import qualified Torch    as T
import qualified Torch.NN as NN


