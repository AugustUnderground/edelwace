{-# OPTIONS_GHC -Wall #-}

module PER 
    ( ReplayBuffer (..)
    , makeBuffer
    , bufferSample
    , bufferRandomSample
    , bufferPush
    , bufferLength
    , PERBuffer (..)
    , makePERBuffer
    , perPush
    , perPush'
    , perSample 
    , perUpdate
    , betaByFrame
    ) where

import Lib

import qualified Torch as T
import qualified Torch.Functional.Internal as T (indexAdd)

------------------------------------------------------------------------------
-- Replay Buffer
------------------------------------------------------------------------------

-- | Strict Simple/Naive Replay Buffer
data ReplayBuffer = ReplayBuffer { states  :: !T.Tensor
                                 , actions :: !T.Tensor
                                 , rewards :: !T.Tensor
                                 , states' :: !T.Tensor
                                 , dones   :: !T.Tensor
                                 } deriving (Show, Eq)

-- | Create a new, empty Buffer on the CPU
makeBuffer :: ReplayBuffer
makeBuffer = ReplayBuffer ft ft ft ft bt
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | Sorta like fmap but specifically for Replay Buffers
bmap :: (T.Tensor -> T.Tensor) -> ReplayBuffer -> ReplayBuffer
bmap f (ReplayBuffer s a r s' d) = ReplayBuffer (f s) (f a) (f r) (f s') (f d)

-- | How many Trajectories are currently stored in memory
bufferLength :: ReplayBuffer -> Int
bufferLength = head . T.shape . states

-- | Push new memories into Buffer
bufferPush :: Int -> ReplayBuffer -> T.Tensor -> T.Tensor -> T.Tensor 
                  -> T.Tensor -> T.Tensor -> ReplayBuffer
bufferPush cap (ReplayBuffer s a r n d) s' a' r' n' d' = buf
  where
    dim     = T.Dim 0
    drop' t = T.sliceDim 0 (flip (-) cap . head .T.shape $ t) 
                           (head . T.shape $ t) 1 t
    [s'',a'',r'',n'',d''] = map (drop' . T.cat dim) 
                          $ zipWith (\src tgt -> [src,tgt]) [s,a,r,n,d] 
                                                            [s',a',r',n',d']
    buf     = ReplayBuffer s'' a'' r'' n'' d''

-- | Get the given indices from Buffer
bufferSample :: ReplayBuffer -> T.Tensor -> ReplayBuffer
bufferSample (ReplayBuffer s a r n d) idx = ReplayBuffer s' a' r' n' d'
  where
    [s',a',r',n',d'] = map (T.indexSelect 0 idx) [s,a,r,n,d]

-- | Uniform random sample from Replay Buffer
bufferRandomSample :: ReplayBuffer -> Int -> IO ReplayBuffer
bufferRandomSample buf batchSize = bufferSample buf 
                                <$> T.multinomialIO i' batchSize False
  where
    i' = T.ones' [bufferLength buf]

-- | Strict Prioritized Experience Replay Buffer
data PERBuffer = PERBuffer { memories   :: !ReplayBuffer
                           , priorities :: !T.Tensor
                           , capacity   :: !Int
                           , alpha      :: !Float
                           , betaStart  :: !Float
                           , betaFrames :: !Int 
                           } deriving (Show, Eq)

-- | Create an empty PER Buffer
makePERBuffer :: Int -> Float -> Float -> Int -> PERBuffer
makePERBuffer = PERBuffer buf prio
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    buf  = makeBuffer
    prio = T.asTensor' ([] :: [Float]) opts

-- | Push new memories in a Buffer
perPush :: PERBuffer -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
        -> T.Tensor -> PERBuffer
perPush (PERBuffer m p c a bs bf) s' a' r' n' d' = PERBuffer m' p' c a bs bf
  where
    m' = bufferPush c m s' a' r' n' d'
    p' = (if bufferLength m > 0 then T.max p else 1.0) * T.onesLike r'

-- | Syntactic Sugar for adding one buffer to another
perPush' :: PERBuffer -> PERBuffer -> PERBuffer
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
perSample :: PERBuffer -> Int -> Int -> IO (ReplayBuffer, T.Tensor, T.Tensor)
perSample (PERBuffer m p _ _ bs bf) frameIdx batchSize = do
    i <- T.toDevice gpu <$> T.multinomialIO p' batchSize False
    let s = bmap (T.indexSelect 0 i) m
        w_ = T.pow (- b) (n * T.indexSelect 0 i p')
        w = w_ / T.max w_
    return (s, i, w)
  where
    n = realToFrac $ bufferLength m
    p'' = T.pow (2.0 :: Float) p
    p' = T.squeezeAll $ p'' / T.sumAll p''
    b = betaByFrame bs bf frameIdx

-- | Update the Priorities of a Buffer
perUpdate :: PERBuffer -> T.Tensor -> T.Tensor -> PERBuffer
perUpdate (PERBuffer m p c a bs bf) idx prio = buf'
  where
    p' = T.indexAdd (T.indexAdd p 0 idx (-1.0 * T.indexSelect 0 idx p)) 0 idx prio
    buf' = PERBuffer m p' c a bs bf
