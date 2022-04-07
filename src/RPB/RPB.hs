{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.RPB ( ReplayBuffer (..)
               , mkBuffer
               , bufferLength
               , bufferPush
               , bufferPush'
               , bufferPop
               , bufferSample
               , bufferRandomSample
               ) where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
-- Replay Buffer
------------------------------------------------------------------------------

-- | Strict Simple/Naive Replay Buffer
data ReplayBuffer a = ReplayBuffer { rpbStates  :: !a   -- ^ States
                                   , rpbActions :: !a   -- ^ Actions
                                   , rpbRewards :: !a   -- ^ Rewards
                                   , rpbStates' :: !a   -- ^ Next States
                                   , rpbDones   :: !a   -- ^ Terminal Mask
                                   } deriving (Show, Eq)

instance Functor ReplayBuffer where
  fmap f (ReplayBuffer s a r s' d) = ReplayBuffer (f s) (f a) (f r) (f s') (f d)

-- | Create a new, empty Buffer on the GPU
mkBuffer :: ReplayBuffer T.Tensor
mkBuffer = ReplayBuffer ft ft ft ft bt
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | How many Trajectories are currently stored in memory
bufferLength :: ReplayBuffer T.Tensor -> Int
bufferLength = head . T.shape . rpbStates

-- | Drop number of entries from the beginning of the Buffer
bufferDrop :: Int -> ReplayBuffer T.Tensor -> ReplayBuffer T.Tensor
bufferDrop cap buf = fmap (T.indexSelect 0 idx) buf
  where
    opts  = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    len   = bufferLength buf
    idx'  = if len < cap
               then ([0 .. (len - 1)] :: [Int])
               else ([(len - cap) .. (len - 1)] :: [Int])
    idx   = T.asTensor' idx' opts

-- | Push new memories into Buffer
bufferPush :: Int -> ReplayBuffer T.Tensor-> T.Tensor -> T.Tensor -> T.Tensor 
           -> T.Tensor -> T.Tensor -> ReplayBuffer T.Tensor
bufferPush cap (ReplayBuffer s a r n d) s' a' r' n' d' = buf
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    r'' = T.cat dim [r, r']
    n'' = T.cat dim [n, n']
    d'' = T.cat dim [d, d']
    buf = bufferDrop cap (ReplayBuffer s'' a'' r'' n'' d'')

-- | Push one buffer into another one
bufferPush' :: Int -> ReplayBuffer T.Tensor -> ReplayBuffer T.Tensor 
            -> ReplayBuffer T.Tensor
bufferPush' cap buf (ReplayBuffer s a r n d) = bufferPush cap buf s a r n d

-- | Pop numElems from Buffer
bufferPop :: Int -> ReplayBuffer T.Tensor -> ReplayBuffer T.Tensor
bufferPop numElems buf = bufferSample idx buf
  where
    bs  = bufferLength buf
    idx = toIntTensor ([(bs - numElems) .. (bs - 1)] :: [Int])

-- | Get the given indices from Buffer
bufferSample :: T.Tensor -> ReplayBuffer T.Tensor -> ReplayBuffer T.Tensor
bufferSample idx = fmap (T.indexSelect 0 idx)

-- | Uniform random sample from Replay Buffer
bufferRandomSample :: Int -> ReplayBuffer T.Tensor -> IO (ReplayBuffer T.Tensor)
bufferRandomSample batchSize buf = (`bufferSample` buf)
                                <$> T.multinomialIO i' batchSize False
  where
    i' = toFloatGPU $ T.ones' [bufferLength buf]
