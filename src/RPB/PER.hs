{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.PER ( PERBuffer (..)
               , mkPERBuffer
               , perPush
               , perPush'
               , perSample 
               , perUpdate
               , betaByFrame
               ) where

import Lib
import RPB.RPB

import qualified Torch as T
import qualified Torch.Functional.Internal as T (indexAdd)

------------------------------------------------------------------------------
-- Prioritized Experience Replay
------------------------------------------------------------------------------

-- | Strict Prioritized Experience Replay Buffer
data PERBuffer a = PERBuffer { perMemories   :: ReplayBuffer a -- ^ Actual Buffer
                             , perPriorities :: !T.Tensor      -- ^ Sample Weights
                             , perCapacity   :: !Int           -- ^ Buffer Capacity
                             , perAlpha      :: !Float         -- ^ Exponent Alpha
                             , perBetaStart  :: !Float         -- ^ Initial Exponent Beta
                             , perBetaFrames :: !Int           -- ^ Beta Decay
                             } deriving (Show, Eq)

instance Functor PERBuffer where
  fmap f (PERBuffer m p c a bs bf) = PERBuffer (fmap f m) p c a bs bf

-- | Create an empty PER Buffer
mkPERBuffer :: Int -> Float -> Float -> Int -> PERBuffer T.Tensor
mkPERBuffer = PERBuffer buf prio
  where
    buf  = mkBuffer
    prio = emptyTensor

-- | Push new memories in a Buffer
perPush :: PERBuffer T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
        -> T.Tensor -> PERBuffer T.Tensor
perPush (PERBuffer m p c a bs bf) s' a' r' n' d' = PERBuffer m' p' c a bs bf
  where
    m' = bufferPush c m s' a' r' n' d'
    p' = (if bufferLength m > 0 then T.max p else 1.0) * T.onesLike (rpbRewards m')

-- | Syntactic Sugar for adding one buffer to another
perPush' :: PERBuffer T.Tensor -> PERBuffer T.Tensor -> PERBuffer T.Tensor
perPush' buffer (PERBuffer (ReplayBuffer s a r s' d) _ _ _ _ _) 
    = perPush buffer s a r s' d

-- | Calculate the β exponent at a given frame
betaByFrame :: Float -> Int -> Int -> Float
betaByFrame bs bf frameIdx = min 1.0 b'
  where
    fi = realToFrac frameIdx
    bf' = realToFrac bf
    b' = bs + fi  * (1.0 - bs) / bf'

-- | Take a prioritized sample from the Buffer
perSample :: PERBuffer T.Tensor -> Int -> Int 
          -> IO (ReplayBuffer T.Tensor, T.Tensor, T.Tensor)
perSample (PERBuffer m p _ _ bs bf) frameIdx batchSize = do
    i <- T.toDevice gpu <$> T.multinomialIO p' batchSize False
    let s = fmap (T.indexSelect 0 i) m
        w' = T.pow (- b) (n * T.indexSelect 0 i p')
        w = w' / T.max w'
    return (s, i, w)
  where
    n   = realToFrac $ bufferLength m
    p'' = T.pow (2.0 :: Float) p
    p'  = T.squeezeAll (p'' / T.sumAll p'')
    b   = betaByFrame bs bf frameIdx

-- | Update the Priorities of a Buffer
perUpdate :: PERBuffer T.Tensor -> T.Tensor -> T.Tensor -> PERBuffer T.Tensor
perUpdate (PERBuffer m p c a bs bf) idx prio = buf'
  where
    p' = T.indexAdd (T.indexAdd p 0 idx (-1.0 * T.indexSelect 0 idx p)) 0 idx prio
    buf' = PERBuffer m p' c a bs bf
