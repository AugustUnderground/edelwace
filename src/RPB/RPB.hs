{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Default / Naive Replay Buffer
module RPB.RPB ( Buffer (..)
               , mkBuffer
               , empty
               , size
               , push
               , push'
               , pop
               , sample
               , sampleIO
               , scaleStates
               ) where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
-- Replay Buffer
------------------------------------------------------------------------------

-- | Strict Simple/Naive Replay Buffer
data Buffer a = Buffer { states  :: !a   -- ^ States
                       , actions :: !a   -- ^ Actions
                       , rewards :: !a   -- ^ Rewards
                       , states' :: !a   -- ^ Next States
                       , dones   :: !a   -- ^ Terminal Mask
                       } deriving (Show, Eq)

instance Functor Buffer where
  fmap f (Buffer s a r s' d) = Buffer (f s) (f a) (f r) (f s') (f d)

-- | Create a new, empty Buffer on the GPU
mkBuffer :: Buffer T.Tensor
mkBuffer = Buffer ft ft ft ft bt
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | Create Empty Buffer
empty :: Buffer T.Tensor
empty = mkBuffer

-- | How many Trajectories are currently stored in memory
size :: Buffer T.Tensor -> Int
size = head . T.shape . states

-- | Drop number of entries from the beginning of the Buffer
drop :: Int -> Buffer T.Tensor -> Buffer T.Tensor
drop cap buf = fmap (T.indexSelect 0 idx) buf
  where
    opts  = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    len   = size buf
    idx   = if len < cap
               then T.arange      0      len 1 opts
               else T.arange (len - cap) len 1 opts
-- | Push new memories into Buffer
push :: Int -> Buffer T.Tensor-> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
     -> T.Tensor -> Buffer T.Tensor
push cap (Buffer s a r n d) s' a' r' n' d' = buf
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    r'' = T.cat dim [r, r']
    n'' = T.cat dim [n, n']
    d'' = T.cat dim [d, d']
    buf = RPB.RPB.drop cap (Buffer s'' a'' r'' n'' d'')

-- | Push one buffer into another one
push' :: Int -> Buffer T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
push' cap buf (Buffer s a r n d) = push cap buf s a r n d

-- | Pop numElems from Buffer
pop :: Int -> Buffer T.Tensor -> Buffer T.Tensor
pop numElems buf = sample idx buf
  where
    bs  = size buf
    idx = toIntTensor ([(bs - numElems) .. (bs - 1)] :: [Int])

-- | Get the given indices from Buffer
sample :: T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
sample idx = fmap (T.indexSelect 0 idx)

-- | Uniform random sample from Replay Buffer
sampleIO :: Int -> Buffer T.Tensor -> IO (Buffer T.Tensor)
sampleIO batchSize buf = (`sample` buf)
                                <$> T.multinomialIO i' batchSize False
  where
    i' = toFloatGPU $ T.ones' [size buf]

-- | Scale and clip states and states'
scaleStates :: Float -> Buffer T.Tensor -> Buffer T.Tensor 
scaleStates c Buffer{..} = buf'
  where
    scaledStates  = T.clamp (- c) c $ rescale states
    scaledStates' = T.clamp (- c) c $ rescale states'
    buf'          = Buffer scaledStates actions rewards scaledStates' dones
