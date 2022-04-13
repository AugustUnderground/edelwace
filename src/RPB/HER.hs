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
               , sample
               , sampleTargets
               , newReward
               , augmentTarget'
               , augmentTarget
               , asRPB
               ) where

import Lib
import qualified RPB.RPB                   as RPB

import Control.Monad
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (where')

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
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | Create an empty HER Buffer
empty :: Buffer T.Tensor
empty = mkBuffer

-- | How many Trajectories are currently stored in memory
size :: Buffer T.Tensor -> Int
size = head . T.shape . states

-- | Drop everything after done (used for single episode)
drop' :: Buffer T.Tensor -> Buffer T.Tensor
drop' buf@Buffer{..} = buf'
  where
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    nz   = T.squeezeAll . T.nonzero . T.flattenAll $ dones
    idx  = T.arange 0 ((T.asValue nz :: Int) + 1) 1 opts
    buf' = if null $ T.shape nz 
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

-- | Calculate reward for new targets given a relative tolerance
newReward :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
newReward relTol obs tgt = rew'
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    msk  = T.allDim (T.Dim 1) True $ T.le (T.abs (obs - tgt) / tgt) relTol
    dims = [head $ T.shape obs, 1]
    rew' = T.where' msk (T.full dims (  0.0  :: Float) opts)
                        (T.full dims ((-1.0) :: Float) opts) 

-- | Augment by changing acheived target, done flag and reward
augmentTarget' :: T.Tensor -> T.Tensor -> Buffer T.Tensor -> Buffer T.Tensor
augmentTarget' tol tgt (Buffer s a _ n _ _ t') = Buffer s a r' n d' tgt tgt
  where
    r' = newReward tol t' tgt
    d' = T.ge r' $ toTensor (0.0 :: Float)

-- | Augment targets for given indices
augmentTarget :: Int -> T.Tensor -> [T.Tensor] -> Buffer T.Tensor 
              -> Buffer T.Tensor
augmentTarget k tol idx buf = buf'
  where
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
    tgt' = T.cat (T.Dim 0) 
         $ map (\i -> T.indexSelect 0 i $ targets' buf) idx
    len  = head $ T.shape tgt'
    idx' = T.arange 0 (size buf - k) 1 opts
    buf' = augmentTarget' tol tgt'
         $ T.reshape [len, -1] . T.repeat [1,k] . T.indexSelect 0 idx' <$> buf

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
    rewards' = newReward tol targets' tgt'
    dones'   = T.ge rewards' (toTensor (0.0 :: Float))
    buf'     = Buffer states actions rewards' states' dones' tgt' tgt'
    cap      = bs + size buf'
sampleTargets Episode k tol buf = do 
    idx      <-  map T.squeezeAll .  T.split 1 (T.Dim 0) 
             <$> T.randintIO 0 bs [bs, k] opts
    let buf' = augmentTarget k tol idx buf
        cap  = bs + size buf'
    pure     $ push' cap buf buf'
  where
    bs       = size buf
    opts     = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts
sampleTargets Random k tol buf = sampleTargets Episode k tol buf
sampleTargets Future k tol buf = do 
    buf''    <-  sampleTargets Final k tol 
              $  T.indexSelect 0 (T.arange (bs - k)    bs    1 opts) 
             <$> buf
    idx      <- forM [0 .. (bs - k - 1)] (\k' -> T.randintIO k' bs [k] opts)
    let buf' = augmentTarget k tol idx buf
        cap  = bs + size buf' + size buf''
    pure     $ foldl (push' cap) buf [buf', buf'']
  where
    bs       = size buf
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
