{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.ERE ( samplingRange
               , sample
               , anneal
               ) where

import Lib
import qualified RPB.RPB as RPB

import qualified Torch   as T

------------------------------------------------------------------------------
-- Emphasizing Recente Experience
------------------------------------------------------------------------------

-- | Calculate ERE Sampling range cK
samplingRange :: Int    -- ^ Buffer Size N
                 -> Int    -- ^ Number of Epochs K
                 -> Int    -- ^ Current Epoch k
                 -> Int    -- ^ cMin
                 -> Float  -- ^ η
                 -> Int    -- ^ cK
samplingRange n k k' cMin η = cK
  where 
    n'  = realToFrac n  :: Float
    κ   = realToFrac k  :: Float
    κ'  = realToFrac k' :: Float
    cK' = round $ n' * (η ** (κ' * (1000 / κ)))
    cK  = max cK' cMin

-- | Sample for buffer within ERE range
sample :: RPB.Buffer T.Tensor -> Int -> Int -> Int -> Int -> Int -> Float
          -> IO (RPB.Buffer T.Tensor)
sample buf cap bs epochs epoch cMin η =  (`RPB.sample` buf) 
                                        <$> randomInts lo hi bs'
  where
    bl  = RPB.size buf
    bs' = min bs bl
    cK  = samplingRange cap epochs epoch cMin η
    lo  = max 0 (bl - cK)
    hi  = bl - 1

-- | ERE η Annealing during training
anneal :: Float -- ^ Initial η0
          -> Float -- ^ Final ηt
          -> Int   -- ^ Horizon T
          -> Int   -- ^ Current step t
          -> Float -- ^ Current ηt
anneal η0 ηT t t' = ηt
  where
    τ  = realToFrac t  :: Float
    τ' = realToFrac t' :: Float
    ηt = η0 + (ηT - η0) * (τ' / τ)
