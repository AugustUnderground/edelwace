{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
    ( ACEURL
    , Step (..)
    , Info (..)
    , ReplayBuffer (..)
    , sampleBuffer
    , pushBuffer
    , lengthBuffer
    , emptyBuffer
    , toTensor
    , toScalar
    , toFloatGPU
    , toFloatCPU
    , toDoubleGPU
    , mapToTensor
    , tensorToMap
    , aceURL
    , aceGet
    , acePoolReset
    , acePoolObsKeys
    , acePoolRandomAction
    , acePoolRandomStep
    , acePoolStep
    , acePoolTarget
    , randomInts
    , computingDevice
    , dataType
    , resetPool
    , obsKeysPool
    , stepPool
    , stepsToTuple
    , writeLog
    , writeEnvLog
    ) where

import qualified Data.ByteString.Lazy as BS
import Data.ByteString.Lazy.Search (replace)
import qualified Data.Map as M
import Data.Maybe
import Data.Aeson
import Data.Functor
import Network.Wreq
import Control.Lens
import GHC.Generics
import System.Directory
import qualified Torch as T
import qualified Torch.Lens as TL

------------------------------------------------------------------------------
-- Data Conversion
------------------------------------------------------------------------------

-- | Default Tensor Computing Device
computingDevice :: T.Device
computingDevice = T.Device T.CUDA 1

-- | Default Tensor Data Type
dataType :: T.DType
dataType = T.Float

-- | Convert an Array to a Tensor
toTensor :: T.TensorLike a => a -> T.Tensor
toTensor t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice computingDevice $ T.defaultOpts

-- | Convert a Scalar to a Tensor
toScalar :: Float -> T.Tensor
toScalar t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice computingDevice $ T.defaultOpts

-- | Convert model to Double on GPU
toDoubleGPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toDoubleGPU = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice computingDevice . T.toType T.Double

-- | Convert model to Float on CPU
toFloatGPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloatGPU = over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice computingDevice . T.toType T.Float

-- | Convert model to Float on CPU
toFloatCPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloatCPU = over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice computingDevice . T.toType T.Float

randomInts :: Int -> Int -> Int -> IO T.Tensor
randomInts lo hi num = T.randintIO lo hi [num] opts 
  where
    opts = T.withDType T.Int64 . T.withDevice computingDevice $ T.defaultOpts

------------------------------------------------------------------------------
-- Replay Buffer
------------------------------------------------------------------------------

-- Data Type for Storing Trajectories
data ReplayBuffer = ReplayBuffer { states  :: T.Tensor
                                 , actions :: T.Tensor
                                 , rewards :: T.Tensor
                                 , states' :: T.Tensor
                                 , dones   :: T.Tensor
                                 } deriving (Show)

-- Get the given indices from Buffer
sampleBuffer :: ReplayBuffer -> T.Tensor -> ReplayBuffer
sampleBuffer (ReplayBuffer s a r n d) idx = ReplayBuffer s' a' r' n' d'
  where
    [s',a',r',n',d'] = map (T.indexSelect 0 idx) [s,a,r,n,d]

-- Push new memories into Buffer
pushBuffer :: ReplayBuffer -> ReplayBuffer -> ReplayBuffer
pushBuffer (ReplayBuffer s a r n d) (ReplayBuffer s' a' r' n' d') = buf
  where
    dim = T.Dim 0
    [s'',a'',r'',n'',d''] = map (T.cat dim) 
                          $ zipWith (\src tgt -> [src,tgt]) [s,a,r,n,d] 
                                                            [s',a',r',n',d']
    buf = ReplayBuffer s'' a'' r'' n'' d''

-- How many Trajectories are currently stored in memory
lengthBuffer :: ReplayBuffer -> Int
lengthBuffer = head . T.shape . states

-- Create a new, empty Buffer
emptyBuffer :: ReplayBuffer
emptyBuffer = ReplayBuffer ft ft ft ft bt
  where
    ft   = T.asTensor' ([] :: [Float]) opts
    bt   = T.asTensor' ([] :: [Bool]) opts
    opts = T.withDType dataType . T.withDevice computingDevice $ T.defaultOpts

------------------------------------------------------------------------------
-- ACE Backend
------------------------------------------------------------------------------

-- Info object gotten form stepping
data Info = Info { outputParameters :: [String]
                 , inputParameters :: [String] 
                 } deriving (Generic, Show)
instance FromJSON Info
instance ToJSON Info

-- Environment Step
data Step = Step { observation :: [Float]
                 , reward :: Float
                 , done :: Bool
                 , info :: Info 
                 } deriving (Generic, Show)
instance FromJSON Step
instance ToJSON Step

-- Route to GACE Server
type ACEURL = String

-- Generate URL from meta information
aceURL :: String -> String -> String -> String -> String -> ACEURL
aceURL h p i b v = u
  where
    r = i ++ "-" ++ b ++ "-v" ++ v
    u = "http://" ++ h ++ ":" ++ p ++ "/" ++ r

-- Send a GET Request to an ACE Server
aceGet :: ACEURL -> String -> IO BS.ByteString
aceGet url route =  get (url ++ "/" ++ route) 
                <&> replace "output-parameters" ("outputParameters" :: BS.ByteString)
                 .  replace "input-parameters" ("inputParameters" :: BS.ByteString)
                 .  (^. responseBody)

-- Send a POST Request to an ACE Server
acePost :: ACEURL -> String -> Value -> IO BS.ByteString
acePost url route payload = post (url ++ "/" ++ route) payload
                         <&> replace "output-parameters" ("outputParameters" :: BS.ByteString)
                          .  replace "input-parameters" ("inputParameters" :: BS.ByteString)
                          .  (^. responseBody)

-- Convert a JSON Response from an ACE Server to a Map
acePoolMap :: ACEURL -> String -> IO (M.Map Int (M.Map String Float))
acePoolMap url route = aceGet url route <&> fromJust . decode

-- Convert a JSON Response from an ACE Server to a List
acePoolList :: ACEURL -> String -> IO (M.Map Int [Float])
acePoolList url route = aceGet url route <&> fromJust . decode

-- Obtain the Target of Pooled GACE Environments
acePoolTarget :: ACEURL -> IO (M.Map Int (M.Map String Float))
acePoolTarget = flip acePoolMap "target"

-- Reset Pooled GACE Environments
acePoolReset :: ACEURL -> IO (M.Map Int [Float])
acePoolReset = flip acePoolList "reset"

-- Observation Keys
acePoolObsKeys :: ACEURL -> IO (M.Map Int [String])
acePoolObsKeys url = aceGet url "observation_keys" <&> fromJust . decode

-- Get Random Actions from all Pooled Environments
acePoolRandomAction :: ACEURL -> IO (M.Map Int [Float])
acePoolRandomAction = flip acePoolList "random_action"

-- Perform Random Actions in all Pooled Environments
acePoolRandomStep :: ACEURL -> IO (M.Map Int Step)
acePoolRandomStep url = aceGet url "random_step" <&> fromJust . decode

-- Take Steps in All Environments
acePoolStep :: ACEURL -> M.Map Int [Float] -> IO (M.Map Int Step)
acePoolStep url action = fromJust . decode 
                      <$> acePost url "step" (toJSON . M.mapKeys show $ action)

-- Convert a Map to a Tensor where Pool index is a dimension
mapToTensor :: M.Map Int [Float] -> T.Tensor
mapToTensor = toTensor . M.elems

-- Convert Tensor to Map
tensorToMap :: T.Tensor -> M.Map Int [Float]
tensorToMap = M.fromList . zip [0 .. ] . T.asValue

-- Convert the Pooled Step Map to a Tuple
stepsToTuple :: M.Map Int Step -> (T.Tensor, T.Tensor, T.Tensor, [Info])
stepsToTuple steps = (obs, rew, don, inf)
  where
    obs = toTensor . M.elems . M.map observation $ steps
    rew = toTensor . M.elems . M.map reward      $ steps
    don = toTensor . M.elems . M.map done        $ steps
    inf =            M.elems . M.map info        $ steps

-- Reset an Environment
resetPool :: ACEURL -> IO T.Tensor
resetPool url = acePoolReset url <&> mapToTensor

-- Shorthand for getting keys of pooled same envs
obsKeysPool :: ACEURL -> IO [String]
obsKeysPool url = fromJust . M.lookup 0 <$> acePoolObsKeys url 

-- Step in an Environment
stepPool :: ACEURL -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor, [Info])
stepPool url action = acePoolStep url (tensorToMap action) <&> stepsToTuple

------------------------------------------------------------------------------
-- Data Visualization
------------------------------------------------------------------------------

-- Obtain current performance from a gace server and write/append to log
writeEnvLog :: FilePath -> ACEURL -> IO ()
writeEnvLog logPath aceUrl = do
    performance <- M.map (M.map (: [])) <$> acePoolMap aceUrl performanceRoute 
    sizing <- M.map (M.map (: [])) <$> acePoolMap aceUrl sizingRoute 
    let envData = M.unionWith M.union performance sizing
    fex <- doesFileExist logPath
    logData' <- if fex then M.unionWith (M.unionWith (++)) envData 
                                . fromJust . decode <$> BS.readFile logPath
                       else return performance
    BS.writeFile jsonPath (encode logData')
  where
    performanceRoute = "current_performance"
    sizingRoute      = "current_sizing"
    jsonPath         = logPath ++ "/env.json"

-- Write an arbitrary Map to a log file for visualization
writeLog :: FilePath -> M.Map String [Float] -> IO ()
writeLog logPath logData = do
    fex <- doesFileExist logPath
    logData' <- if fex then M.unionWith (++) logData . fromJust . decode 
                            <$> BS.readFile logPath
                       else return logData
    BS.writeFile jsonPath (encode logData')
  where
    jsonPath = logPath ++ "/log.json"
