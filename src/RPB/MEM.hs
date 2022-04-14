{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | PPO Style Replay Memory and Memory Loaders
module RPB.MEM ( Buffer (..)
               , mkBuffer
               , empty
               , size
               , push
               , push'
               , gae
               , Loader (..)
               , mkLoader
               , size'
               ) where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
-- Replay Memory
------------------------------------------------------------------------------

-- | Replay Memory
data Buffer a = Buffer { states   :: !a  -- ^ States
                       , actions  :: !a  -- ^ Action
                       , logProbs :: !a  -- ^ Logarithmic Probability
                       , rewards  :: !a  -- ^ Rewards
                       , values   :: !a  -- ^ Values
                       , masks    :: !a  -- ^ Terminal Mask
                       } deriving (Show, Eq)

instance Functor Buffer where
  fmap f (Buffer s a l r v m) = Buffer (f s) (f a) (f l) (f r) (f v) (f m)

-- | Create a new, empty Buffer on the GPU
mkBuffer :: Buffer T.Tensor
mkBuffer = Buffer ft ft ft ft ft bt
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

-- | Push new memories into Buffer
push :: Buffer T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
           -> T.Tensor -> T.Tensor -> Buffer T.Tensor
push (Buffer s a l r v m) s' a' l' r' v' m' = mem
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    l'' = T.cat dim [l, l']
    r'' = T.cat dim [r, r']
    v'' = T.cat dim [v, v']
    m'' = T.cat dim [m, m']
    mem = Buffer s'' a'' l'' r'' v'' m''

-- | Pushing one buffer into another one
push' :: Buffer T.Tensor -> Buffer T.Tensor 
            -> Buffer T.Tensor
push' mem (Buffer s a l r v m) = push mem s a l r v m

-- | Generalized Advantage Estimate
gae :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
    -> T.Tensor
gae r v m v' γ τ = a
  where
    δ = r + γ * v' * m - v
    l = reverse [0 .. (head . T.shape $ δ) - 1]
    i = toTensor ([0.0] :: [Float])
    z = toIntTensor (0 :: Int)
    gae' :: T.Tensor -> Int -> T.Tensor
    gae' g' i' = T.cat (T.Dim 0) [e, g']
      where  
        δ'  = T.indexSelect 0 (toIntTensor i') δ
        m'  = T.indexSelect 0 (toIntTensor i') m
        g'' = T.indexSelect 0 z g'
        e   = δ' + γ * τ * m' *  g''
    g = T.sliceDim 0 0 (- 1) 1 $ foldl gae' i l
    a = v + g

------------------------------------------------------------------------------
-- Replay Memory Data Loader
------------------------------------------------------------------------------

-- | Memory Data Loader
data Loader a = Loader { states'     :: !a -- ^ States
                       , actions'    :: !a -- ^ Actions
                       , logProbs'   :: !a -- ^ Logarithmic Probabilities
                       , returns'    :: !a -- ^ Returns
                       , advantages' :: !a -- ^ Advantages
                       } deriving (Show, Eq)

instance Functor Loader where
  fmap f (Loader s a l r a') = Loader (f s) (f a) (f l) (f r) (f a')

-- | How many Trajectories are currently stored in memory
size' :: Loader [T.Tensor] -> Int
size' = length . states'

-- | Turn Replay memory into chunked data loader
mkLoader :: Buffer T.Tensor -> Int -> T.Tensor -> T.Tensor 
           -> Loader [T.Tensor]
mkLoader mem bs' γ τ = loader
  where
    len        = size mem
    mem'       = T.sliceDim 0 0 (-1) 1 <$> mem
    values''   = T.squeezeAll $ T.sliceDim 0 1 len 1 (values mem)
    rewards'   = T.squeezeAll $ rewards mem'
    values'    = T.squeezeAll $ values mem'
    masks'     = T.squeezeAll $ masks mem'
    returns    = gae rewards' values' masks' values'' γ τ
    advantages = returns - values'
    bl         = realToFrac len :: Float
    bs         = realToFrac bs' :: Float
    chunks     = (ceiling $ bl / bs) :: Int
    loader     = T.chunk chunks (T.Dim 0) 
              <$> Loader (states mem') (RPB.MEM.actions mem') 
                               (logProbs mem') (T.reshape [-1,1] returns) 
                               (T.reshape[-1,1] advantages)
