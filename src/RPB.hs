{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB ( Buffer (..)
           , ReplayBuffer (..)
           , mkBuffer
           , bufferLength
           , bufferPush
           , bufferPush'
           , bufferPop
           , bufferSample
           , bufferRandomSample
           , ereSamplingRange
           , ereSample
           , ereAnneal
           , PERBuffer (..)
           , mkPERBuffer
           , perPush
           , perPush'
           , perSample 
           , perUpdate
           , betaByFrame
           , ReplayMemory (..)
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
import qualified Torch.Functional.Internal as T (indexAdd)

------------------------------------------------------------------------------
-- What kind of buffer do you want?
------------------------------------------------------------------------------

-- | Indicate Buffer Type
data Buffer = RPB    -- ^ Normal Replay Buffer
            | PER    -- ^ Prioritized Experience Replay
            | MEM    -- ^ PPO Style replay Memory
            | ERE    -- ^ Emphasizing Recent Experience
            | PERERE -- ^ PER + ERE
            deriving (Show, Eq)

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

-- | Create a new, empty Buffer on the CPU
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

------------------------------------------------------------------------------
-- Emphasizing Recente Experience
------------------------------------------------------------------------------

-- | Calculate ERE Sampling range cK
ereSamplingRange :: Int    -- ^ Buffer Size N
                 -> Int    -- ^ Number of Epochs K
                 -> Int    -- ^ Current Epoch k
                 -> Int    -- ^ cMin
                 -> Float  -- ^ η
                 -> Int    -- ^ cK
ereSamplingRange n k k' cMin η = cK
  where 
    n'  = realToFrac n  :: Float
    κ   = realToFrac k  :: Float
    κ'  = realToFrac k' :: Float
    cK' = round $ n' * (η ** (κ' * (1000 / κ)))
    cK  = max cK' cMin

-- | Sample for buffer within ERE range
ereSample :: ReplayBuffer T.Tensor -> Int -> Int -> Int -> Int -> Int -> Float
          -> IO (ReplayBuffer T.Tensor)
ereSample buf cap bs epochs epoch cMin η =  (`bufferSample` buf) 
                                        <$> randomInts lo hi bs
  where
    cK = ereSamplingRange cap epochs epoch cMin η
    lo = cap - cK
    hi = cap - 1

-- | ERE η Annealing during training
ereAnneal :: Float -- ^ Initial η0
          -> Float -- ^ Final ηt
          -> Int   -- ^ Horizon T
          -> Int   -- ^ Current step t
          -> Float -- ^ Current ηt
ereAnneal η0 ηT t t' = ηt
  where
    τ  = realToFrac t  :: Float
    τ' = realToFrac t' :: Float
    ηt = η0 + (ηT - η0) * (τ' / τ)

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

-- | Create a new, empty Buffer on the CPU
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
