{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Prioritized Experience Replay
module RPB.PER ( Buffer (..)
               , mkBuffer
               , push
               , push'
               , sampleIO 
               , update
               , betaByFrame
               ) where

import Lib
import qualified RPB.RPB         as RPB

import qualified Torch as T
import qualified Torch.Functional.Internal as T (indexAdd)

------------------------------------------------------------------------------
-- Prioritized Experience Replay
------------------------------------------------------------------------------

-- | Strict Prioritized Experience Replay Buffer
data Buffer a = Buffer { memories   :: RPB.Buffer a -- ^ Actual Buffer
                       , priorities :: !T.Tensor    -- ^ Sample Weights
                       , capacity   :: !Int         -- ^ Buffer Capacity
                       , alpha      :: !Float       -- ^ Exponent Alpha
                       , betaStart  :: !Float       -- ^ Initial Exponent Beta
                       , betaFrames :: !Int         -- ^ Beta Decay
                       } deriving (Show, Eq)

instance Functor Buffer where
  fmap f (Buffer m p c a bs bf) = Buffer (fmap f m) p c a bs bf

-- | Create an empty PER Buffer
mkBuffer :: Int -> Float -> Float -> Int -> Buffer T.Tensor
mkBuffer = Buffer buf prio
  where
    buf  = RPB.mkBuffer
    prio = emptyTensor

-- | Push new memories in a Buffer
push :: Buffer T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
     -> T.Tensor -> Buffer T.Tensor
push (Buffer m p c a bs bf) s' a' r' n' d' = Buffer m' p' c a bs bf
  where
    m' = RPB.push c m s' a' r' n' d'
    p' = (if RPB.size m > 0 then T.max p else 1.0) * T.onesLike (RPB.rewards m')

-- | Syntactic Sugar for adding one buffer to another
push' :: Buffer T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
push' buffer (Buffer (RPB.Buffer s a r s' d) _ _ _ _ _) 
    = push buffer s a r s' d

-- | Calculate the β exponent at a given frame
betaByFrame :: Float -> Int -> Int -> Float
betaByFrame bs bf frameIdx = min 1.0 b'
  where
    fi = realToFrac frameIdx
    bf' = realToFrac bf
    b' = bs + fi  * (1.0 - bs) / bf'

-- | Take a prioritized sample from the Buffer
sampleIO :: Buffer T.Tensor -> Int -> Int 
          -> IO (RPB.Buffer T.Tensor, T.Tensor, T.Tensor)
sampleIO (Buffer m p _ _ bs bf) frameIdx batchSize = do
    i <- T.toDevice gpu <$> T.multinomialIO p' batchSize False
    let s = fmap (T.indexSelect 0 i) m
        w' = T.pow (- b) (n * T.indexSelect 0 i p')
        w = w' / T.max w'
    return (s, i, w)
  where
    n   = realToFrac $ RPB.size m
    p'' = T.pow (2.0 :: Float) p
    p'  = T.squeezeAll (p'' / T.sumAll p'')
    b   = betaByFrame bs bf frameIdx

-- | Update the Priorities of a Buffer
update :: Buffer T.Tensor -> T.Tensor -> T.Tensor -> Buffer T.Tensor
update (Buffer m p c a bs bf) idx prio = buf'
  where
    p' = T.indexAdd (T.indexAdd p 0 idx (-1.0 * T.indexSelect 0 idx p)) 0 idx prio
    buf' = Buffer m p' c a bs bf
