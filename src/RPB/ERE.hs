{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.ERE ( ereSamplingRange
               , ereSample
               , ereAnneal
               ) where

import Lib
import RPB.RPB

import qualified Torch as T

------------------------------------------------------------------------------
-- Emphasizing Recente Experience
------------------------------------------------------------------------------

-- | Calculate ERE Sampling range cK
ereSamplingRange :: Int    -- ^ Buffer Size N
                 -> Int    -- ^ Number of Epochs K
                 -> Int    -- ^ Current Epoch k
                 -> Int    -- ^ cMin
                 -> Float  -- ^ η
                 -> Int    -- ^ cK
ereSamplingRange n k k' cMin η = cK
  where 
    n'  = realToFrac n  :: Float
    κ   = realToFrac k  :: Float
    κ'  = realToFrac k' :: Float
    cK' = round $ n' * (η ** (κ' * (1000 / κ)))
    cK  = max cK' cMin

-- | Sample for buffer within ERE range
ereSample :: ReplayBuffer T.Tensor -> Int -> Int -> Int -> Int -> Int -> Float
          -> IO (ReplayBuffer T.Tensor)
ereSample buf cap bs epochs epoch cMin η =  (`bufferSample` buf) 
                                        <$> randomInts lo hi bs'
  where
    bl  = bufferLength buf
    bs' = min bs bl
    cK  = ereSamplingRange cap epochs epoch cMin η
    lo  = max 0 (bl - cK)
    hi  = bl - 1

-- | ERE η Annealing during training
ereAnneal :: Float -- ^ Initial η0
          -> Float -- ^ Final ηt
          -> Int   -- ^ Horizon T
          -> Int   -- ^ Current step t
          -> Float -- ^ Current ηt
ereAnneal η0 ηT t t' = ηt
  where
    τ  = realToFrac t  :: Float
    τ' = realToFrac t' :: Float
    ηt = η0 + (ηT - η0) * (τ' / τ)


