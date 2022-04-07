{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.HER ( HERBuffer (..)
               , mkHERBuffer
               , herLength
               , herDrop
               , herPush
               , herPush'
               , herSample
               , herRandomSample
               ) where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
-- Hindsight Experience Replay
------------------------------------------------------------------------------

-- | Strict Hindsight Experience Replay Buffer
data HERBuffer a = HERBuffer { herStates  :: !a   -- ^ States
                             , herActions :: !a   -- ^ Actions
                             , herRewards :: !a   -- ^ Rewards
                             , herStates' :: !a   -- ^ Next States
                             , herDones   :: !a   -- ^ Terminal Mask
                             , herTargets :: !a   -- ^ Targets
                             } deriving (Show, Eq)

instance Functor HERBuffer where
  fmap f (HERBuffer s a r s' d t) = HERBuffer (f s) (f a) (f r) (f s') (f d) (f t)

-- | Create a new, empty HER Buffer on the GPU
mkHERBuffer :: HERBuffer T.Tensor
mkHERBuffer = HERBuffer ft ft ft ft bt ft
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | How many Trajectories are currently stored in memory
herLength :: HERBuffer T.Tensor -> Int
herLength = head . T.shape . herStates

-- | Drop number of entries from the beginning of the Buffer
herDrop :: Int -> HERBuffer T.Tensor -> HERBuffer T.Tensor
herDrop cap buf = fmap (T.indexSelect 0 idx) buf
  where
    opts  = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    len   = herLength buf
    idx'  = if len < cap
               then ([0 .. (len - 1)] :: [Int])
               else ([(len - cap) .. (len - 1)] :: [Int])
    idx   = T.asTensor' idx' opts

-- | Push new memories into Buffer
herPush :: Int -> HERBuffer T.Tensor-> T.Tensor -> T.Tensor -> T.Tensor 
        -> T.Tensor -> T.Tensor -> T.Tensor -> HERBuffer T.Tensor
herPush cap (HERBuffer s a r n d t) s' a' r' n' d' t' = buf
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    r'' = T.cat dim [r, r']
    n'' = T.cat dim [n, n']
    d'' = T.cat dim [d, d']
    t'' = T.cat dim [t, t']
    buf = herDrop cap (HERBuffer s'' a'' r'' n'' d'' t'')

-- | Push one buffer into another one
herPush' :: Int -> HERBuffer T.Tensor -> HERBuffer T.Tensor 
            -> HERBuffer T.Tensor
herPush' cap buf (HERBuffer s a r n d t) = herPush cap buf s a r n d t

-- | Get the given indices from Buffer
herSample :: T.Tensor -> HERBuffer T.Tensor -> HERBuffer T.Tensor
herSample idx = fmap (T.indexSelect 0 idx)

-- | Uniform random sample from Replay Buffer
herRandomSample :: Int -> HERBuffer T.Tensor -> IO (HERBuffer T.Tensor)
herRandomSample batchSize buf = (`herSample` buf)
                                <$> T.multinomialIO i' batchSize False
  where
    i' = toFloatGPU $ T.ones' [herLength buf]
