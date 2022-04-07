{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB.MEM ( ReplayMemory (..)
               , mkMemory
               , memoryLength
               , memoryPush
               , memoryPush'
               , gae
               , MemoryLoader (..)
               , dataLoader
               , loaderLength
               ) where

import Lib

import qualified Torch as T

------------------------------------------------------------------------------
-- Replay Memory
------------------------------------------------------------------------------

-- | Replay Memory
data ReplayMemory a = ReplayMemory { memStates   :: !a  -- ^ States
                                   , memActions  :: !a  -- ^ Action
                                   , memLogPorbs :: !a  -- ^ Logarithmic Probability
                                   , memRewards  :: !a  -- ^ Rewards
                                   , memValues   :: !a  -- ^ Values
                                   , memMasks    :: !a  -- ^ Terminal Mask
                                   } deriving (Show, Eq)

instance Functor ReplayMemory where
  fmap f (ReplayMemory s a l r v m) = ReplayMemory (f s) (f a) (f l) (f r) (f v) (f m)

-- | Create a new, empty Buffer on the GPU
mkMemory :: ReplayMemory T.Tensor
mkMemory = ReplayMemory ft ft ft ft ft bt
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts

-- | How many Trajectories are currently stored in memory
memoryLength :: ReplayMemory T.Tensor -> Int
memoryLength = head . T.shape . memStates

-- | Push new memories into Buffer
memoryPush :: ReplayMemory T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
           -> T.Tensor -> T.Tensor -> ReplayMemory T.Tensor
memoryPush (ReplayMemory s a l r v m) s' a' l' r' v' m' = mem
  where
    dim = T.Dim 0
    s'' = T.cat dim [s, s']
    a'' = T.cat dim [a, a']
    l'' = T.cat dim [l, l']
    r'' = T.cat dim [r, r']
    v'' = T.cat dim [v, v']
    m'' = T.cat dim [m, m']
    mem = ReplayMemory s'' a'' l'' r'' v'' m''

-- | Pushing one buffer into another one
memoryPush' :: ReplayMemory T.Tensor -> ReplayMemory T.Tensor 
            -> ReplayMemory T.Tensor
memoryPush' mem (ReplayMemory s a l r v m) = memoryPush mem s a l r v m

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
data MemoryLoader a = MemoryLoader { loaderStates     :: !a -- ^ States
                                   , loaderActions    :: !a -- ^ Actions
                                   , loaderLogPorbs   :: !a -- ^ Logarithmic Probabilities
                                   , loaderReturns    :: !a -- ^ Returns
                                   , loaderAdvantages :: !a -- ^ Advantages
                                   } deriving (Show, Eq)

instance Functor MemoryLoader where
  fmap f (MemoryLoader s a l r a') = MemoryLoader (f s) (f a) (f l) (f r) (f a')

-- | How many Trajectories are currently stored in memory
loaderLength :: MemoryLoader [T.Tensor] -> Int
loaderLength = length . loaderStates

-- | Turn Replay memory into chunked data loader
dataLoader :: ReplayMemory T.Tensor -> Int -> T.Tensor -> T.Tensor 
           -> MemoryLoader [T.Tensor]
dataLoader mem bs' γ τ = loader
  where
    len        = memoryLength mem
    mem'       = T.sliceDim 0 0 (-1) 1 <$> mem
    values'    = T.squeezeAll $ T.sliceDim 0 1 len 1 (memValues mem)
    rewards    = T.squeezeAll $ memRewards mem'
    values     = T.squeezeAll $ memValues mem'
    masks      = T.squeezeAll $ memMasks mem'
    returns    = gae rewards values masks values' γ τ
    advantages = returns - values
    bl         = realToFrac len :: Float
    bs         = realToFrac bs' :: Float
    chunks     = (ceiling $ bl / bs) :: Int
    loader     = T.chunk chunks (T.Dim 0) 
              <$> MemoryLoader (memStates mem') (memActions mem') 
                               (memLogPorbs mem') (T.reshape [-1,1] returns) 
                               (T.reshape[-1,1] advantages)
