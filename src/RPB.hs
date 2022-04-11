{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Replay Buffers and Memory Loaders
module RPB where

------------------------------------------------------------------------------
-- What kind of buffer do you want?
------------------------------------------------------------------------------

-- | Indicate Buffer Type to be used by Algorithm
data BufferType = RPB    -- ^ Normal Replay Buffer (SAC, TD3)
                | PER    -- ^ Prioritized Experience Replay (SAC)
                | MEM    -- ^ PPO Style replay Memory (PPO)
                | ERE    -- ^ Emphasizing Recent Experience (SAC)
                | HER    -- ^ Hindsight Experience Replay (TD3)
                deriving (Show, Eq)
