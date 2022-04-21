{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Hindsight Experience Replay
module RPB.HER ( Strategy (..)
               , Buffer (..) 
               , mkBuffer
               , empty
               , size
               , push
               , push'
               , drop'
               , envSplit
               , epsSplit
               , sample
               , sampleTargets
               , newReward
               , augmentTarget
               , asRPB
               ) where

import Lib
import qualified RPB.RPB                   as RPB

import Control.Monad
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (negative)

------------------------------------------------------------------------------
-- Hindsight Experience Replay
------------------------------------------------------------------------------

-- | Hindsight Experience Replay Strategies for choosing Goals
data Strategy = Final   -- ^ Only Final States are additional targets
              | Random  -- ^ Replay with `k` random states encountered so far (basically RPB)
              | Episode -- ^ Replay with `k` random states from same episode.
              | Future  -- ^ Replay with `k` random states from same episode, that were observed after
  deriving (Show, Eq)

-- | Strict Simple/Naive Replay Buffer
data Buffer a = Buffer { states   :: !a   -- ^ States
                       , actions  :: !a   -- ^ Actions
                       , rewards  :: !a   -- ^ Rewards
                       , states'  :: !a   -- ^ Next States
                       , dones    :: !a   -- ^ Terminal Mask
                       , targets  :: !a   -- ^ Actual Episode Target
                       , targets' :: !a   -- ^ Augmented Target
                       } deriving (Show, Eq)

instance Functor Buffer where
  fmap f (Buffer s a r s' d t t') = 
         Buffer (f s) (f a) (f r) (f s') (f d) (f t) (f t')

-- | Create a new, empty HER Buffer on the GPU
mkBuffer :: Buffer T.Tensor
mkBuffer = Buffer ft ft ft ft bt ft ft
  where
    fopts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    bopts = T.withDType T.Float  . T.withDevice gpu $ T.defaultOpts
    ft    = T.asTensor' ([] :: [Float]) fopts
    bt    = T.asTensor' ([] :: [Bool])  bopts

-- | Create an empty HER Buffer
empty :: Buffer T.Tensor
empty = mkBuffer

-- | How many Trajectories are currently stored in memory
size :: Buffer T.Tensor -> Int
size = head . T.shape . states

-- | Drop everything after last done (used for single episode)
drop' :: Buffer T.Tensor -> Buffer T.Tensor
drop' buf@Buffer{..} = buf'
  where
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    nz   = T.squeezeAll . T.nonzero . T.flattenAll $ dones
    idx  = T.arange 0 (succ (T.asValue nz :: Int)) 1 opts
    buf' = if T.shape nz /= [0]
              then T.indexSelect 0 idx <$> buf
              else buf

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
     -> T.Tensor -> T.Tensor -> T.Tensor -> Buffer T.Tensor
push cap (Buffer s a r n d t g) s' a' r' n' d' t' g' = RPB.HER.drop cap buf
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    r'' = T.cat dim [r, r']
    n'' = T.cat dim [n, n']
    d'' = T.cat dim [d, d']
    t'' = T.cat dim [t, t']
    g'' = T.cat dim [g, g']
    buf = Buffer s'' a'' r'' n'' d'' t'' g''

-- | Push one buffer into another one
push' :: Int -> Buffer T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
push' cap buf (Buffer s a r s' d t t') = push cap buf s a r s' d t t'

-- | Get the given indices from Buffer
sample :: T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
sample idx = fmap (T.indexSelect 0 idx)

-- | Split buffer collected from pool by env
envSplit :: Int -> Buffer T.Tensor -> [Buffer T.Tensor]
envSplit ne buf = map (`sample` buf) idx
  where
    bl   = size buf
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    idx  = map T.squeezeAll . T.split 1 (T.Dim 0) 
         $ T.reshape [-1,1] (T.arange 0 ne 1  opts) + T.arange 0 bl ne opts

-- | Split a buffer into episodes, dropping the last unfinished
epsSplit :: Buffer T.Tensor -> [Buffer T.Tensor]
epsSplit buf@Buffer{..} = map (\i -> fmap (T.indexSelect 0 i) buf) d
  where
    d'' = T.reshape [-1] . T.squeezeAll . T.nonzero . T.squeezeAll $ dones
    d' = T.asValue d'' :: [Int]
    d  = splits' (0:d')

-- | Calculate reward for new targets given a relative tolerance
newReward :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
newReward relTol obs tgt = T.negative . T.toDType T.Float 
                         . T.allDim (T.Dim 1) True . T.gt relTol 
                         . (`T.div` tgt) . T.abs $ T.sub obs tgt

-- | Augment by changing acheived target, done flag and reward
augmentTarget :: T.Tensor -> T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
augmentTarget tol t (Buffer s a _ s' _ _ t') = buf 
  where 
    r' = newReward tol t' t
    d' = T.ge r' $ toTensor (0.0 :: Float)
    buf = Buffer s a r' s' d' t t'

-- | Sample Additional Goals according to Strategy (drop first). `Random` is
-- basically the same as `Episode` you just have to give it the entire buffer,
-- not just the episode.
sampleTargets :: Strategy -> Int -> T.Tensor -> Buffer T.Tensor 
              -> IO (Buffer T.Tensor)
sampleTargets Final _ tol buf@Buffer{..} = pure $ push' cap buf buf'
  where
    bs       = size buf
    idx      = toIntTensor . (:[]) . pred . head . T.shape $ targets'
    tgt'     = T.repeat [bs, 1] $ T.indexSelect 0 idx targets'
    buf'     = augmentTarget tol tgt' buf
    cap      = bs + size buf'
sampleTargets Episode k tol buf = do 
    idx      <-  map T.squeezeAll .  T.split 1 (T.Dim 0) 
             <$> T.randintIO 0 bs [bs, k] opts
    let tgt' = T.cat (T.Dim 0) 
             $ map (\i -> T.indexSelect 0 i $ targets' buf) idx
        rep  = T.full [bs] k opts
        buf' = augmentTarget tol tgt' 
             $ fmap (repeatInterleave' 0 rep) buf
        cap  = bs + size buf'
    pure     $ push' cap buf buf'
  where
    bs       = size buf
    opts     = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
sampleTargets Random k tol buf = sampleTargets Episode k tol buf
sampleTargets Future k tol buf | k >= size buf = pure buf
                               | otherwise     = do 
    buf''    <- sampleTargets Final k tol $ fmap (T.indexSelect 0 idx'') buf
    idx      <- forM [0 .. (bs - k - 1)] (\k' -> T.randintIO k' bs [k] opts)
    let tgt' = T.cat (T.Dim 0) 
             $ map (\i -> T.indexSelect 0 i $ targets' buf) idx
        rep  = T.full [bs - k] k opts
        buf' = augmentTarget tol tgt' 
             $ fmap (repeatInterleave' 0 rep . T.indexSelect 0 idx') buf
        cap  = bs + size buf' + size buf''
    pure     $ foldl (push' cap) buf [buf', buf'']
  where
    bs       = size buf
    idx''    = T.arange (bs - k)    bs   1 opts
    idx'     = T.arange     0   (bs - k) 1 opts
    opts     = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts

-- | Convert HER Buffer to RPB for training
asRPB :: Buffer T.Tensor -> RPB.Buffer T.Tensor
asRPB Buffer{..} = RPB.Buffer s a r n d
  where
    s = T.cat (T.Dim 1) [states, targets]
    a = actions
    r = rewards
    n = T.cat (T.Dim 1) [states', targets]
    d = dones
