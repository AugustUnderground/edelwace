{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib where

import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString      as BS
import qualified Data.Map as M
import Data.Char (isLower)
import Data.List (isInfixOf, isPrefixOf, isSuffixOf, elemIndex)
import Data.Maybe (fromJust)
import Data.Aeson
import Network.Wreq
import Control.Lens
import GHC.Generics
import System.Directory
import qualified Torch as T
import qualified Torch.Lens as TL
import qualified Torch.Functional.Internal as T (where')

------------------------------------------------------------------------------
-- Convenience / Syntactic Sugar
------------------------------------------------------------------------------

-- | Swaps the arguments of HaskTorch's foldLoop around
foldLoop' :: Int -> (a -> Int -> IO a) -> a -> IO a
foldLoop' i f m = T.foldLoop m i f

------------------------------------------------------------------------------
-- Data Conversion
------------------------------------------------------------------------------

-- | GPU 1
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU 0
cpu :: T.Device
cpu = T.Device T.CPU 0

-- | Default Tensor Data Type
dataType :: T.DType
dataType = T.Float

-- | Convert an Array to a Tensor
toTensor :: T.TensorLike a => a -> T.Tensor
toTensor t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Create an empty Float Tensor on GPU
emptyTensor :: T.Tensor
emptyTensor = T.asTensor' ([] :: [Float]) opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Convert a Scalar to a Tensor
toScalar :: Float -> T.Tensor
toScalar t = T.asTensor' t opts
  where
    opts = T.withDType dataType . T.withDevice gpu $ T.defaultOpts

-- | Convert model to Double on GPU
toDoubleGPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toDoubleGPU = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice gpu . T.toType T.Double

-- | Convert model to Float on CPU
toFloatGPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloatGPU = TL.over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice gpu . T.toType T.Float

-- | Convert model to Float on CPU
toFloatCPU :: forall a. TL.HasTypes a T.Tensor => a -> a
toFloatCPU = over (TL.types @ T.Tensor @a) opts
  where
    opts = T.toDevice gpu . T.toType T.Float

-- | Generate a Tensor of random Integers
randomInts :: Int -> Int -> Int -> IO T.Tensor
randomInts lo hi num = T.randintIO lo hi [num] opts 
  where
    opts = T.withDType T.Int64 . T.withDevice gpu $ T.defaultOpts

------------------------------------------------------------------------------
-- Hym Server Interaction
------------------------------------------------------------------------------

-- | Info object gotten form stepping
data Info = Info { observations :: ![String]
                 , actions      :: ![String] 
                 } deriving (Generic, Show)
instance FromJSON Info
instance ToJSON Info

-- | Environment Step
data Step = Step { observation  :: ![Float]
                 , reward       :: !Float
                 , done         :: !Bool
                 , info         :: !Info 
                 } deriving (Generic, Show)
instance FromJSON Step
instance ToJSON Step

-- | Base Route to Hym Server
type HymURL = String

-- | Convert a Map to a Tensor where Pool index is a dimension
mapToTensor :: M.Map Int [Float] -> T.Tensor
mapToTensor = toTensor . M.elems

-- | Convert Tensor to Map
tensorToMap :: T.Tensor -> M.Map Int [Float]
tensorToMap = M.fromList . zip [0 .. ] . T.asValue

-- | Convert the Pooled Step Map to a Tuple
stepsToTuple :: M.Map Int Step -> (T.Tensor, T.Tensor, T.Tensor, [Info])
stepsToTuple steps = (obs, rew, don, inf)
  where
    obs =                    toTensor . M.elems . M.map observation $ steps
    rew = T.reshape [-1,1] . toTensor . M.elems . M.map reward      $ steps
    don = T.reshape [-1,1] . toTensor . M.elems . M.map done        $ steps
    inf =                               M.elems . M.map info        $ steps

-- | Generic HTTP GET Request to Hym Server
hymGet :: HymURL -> String -> IO BS.ByteString
hymGet url route = BL.toStrict . (^. responseBody) <$>  get (url ++ "/" ++ route)

-- | Send a POST Request to a Hym Server
hymPost :: HymURL -> String -> Value -> IO BS.ByteString
hymPost url route payload = BL.toStrict . (^. responseBody) 
                         <$> post (url ++ "/" ++ route) payload 

-- | Convert a JSON Response from an ACE Server to a Map
hymPoolMap :: HymURL -> String -> IO (M.Map Int (M.Map String Float))
hymPoolMap url route = fromJust . decodeStrict <$> hymGet url route

-- | Convert a JSON Response from an ACE Server to a List
hymPoolList :: HymURL -> String -> IO (M.Map Int [Float])
hymPoolList url route = fromJust . decodeStrict <$> hymGet url route

-- | Reset Pooled Environments on a Hym server
hymPoolReset :: HymURL -> IO (M.Map Int [Float])
hymPoolReset = flip hymPoolList "reset"

-- | Get Random Actions from all Pooled Environments
hymPoolRandomAction :: HymURL -> IO (M.Map Int [Float])
hymPoolRandomAction = flip hymPoolList "random_action"

-- | Perform Random Actions in all Pooled Environments
hymPoolRandomStep :: HymURL -> IO (M.Map Int Step)
hymPoolRandomStep url = fromJust . decodeStrict <$> hymGet url "random_step"

-- | Take Steps in All Environments
hymPoolStep :: HymURL -> M.Map Int [Float] -> IO (M.Map Int Step)
hymPoolStep url action = fromJust . decodeStrict
                      <$> hymPost url "step" (toJSON . M.mapKeys show $ action)

-- | Generate URL to a Hym-GACE server from meta information
aceURL :: String -> String -> String -> String -> String -> HymURL
aceURL h p i b v = "http://" ++ h ++ ":" ++ p ++ "/" ++ i ++ "-" ++ b ++ "-v" ++ v

-- | Generate URL to a Hym-Gym server from meta information
gymURL :: String -> String -> String -> String -> HymURL
gymURL h p i v = "http://" ++ h ++ ":" ++ p ++ "/" ++ i ++ "-v" ++ v

-- | Send a GET Request to a GACE Server
-- Obtain the Target of Pooled GACE Environments
acePoolTarget :: HymURL -> IO (M.Map Int (M.Map String Float))
acePoolTarget = flip hymPoolMap "target"

-- | Action Keys from GACE Server
acePoolActKeys :: HymURL -> IO (M.Map Int [String])
acePoolActKeys url = fromJust . decodeStrict <$> hymGet url "action_keys"

-- | Observation Keys from GACE Server
acePoolObsKeys :: HymURL -> IO (M.Map Int [String])
acePoolObsKeys url = fromJust . decodeStrict <$> hymGet url "observation_keys"

-- | Reset a Vectorized Environment Pool
resetPool :: HymURL -> IO T.Tensor
resetPool url = mapToTensor <$> hymPoolReset url

-- | Shorthand for getting keys of pooled same envs
actKeysPool :: HymURL -> IO [String]
actKeysPool url = fromJust . M.lookup 0 <$> acePoolActKeys url 

-- | Shorthand for getting keys of pooled same envs
obsKeysPool :: HymURL -> IO [String]
obsKeysPool url = fromJust . M.lookup 0 <$> acePoolObsKeys url 

-- | Get Info without stepping
infoPool :: HymURL -> IO Info
infoPool url = do
    obs <- obsKeysPool url
    act <- actKeysPool url
    return (Info obs act)

-- | Step in an Environment
stepPool :: HymURL -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor, [Info])
stepPool url action = stepsToTuple <$> hymPoolStep url (tensorToMap action)

------------------------------------------------------------------------------
-- Data Processing
------------------------------------------------------------------------------

-- | Process / Sanitize the Observations from GACE
processGace :: T.Tensor -> Info -> T.Tensor
processGace obs Info {..} = states
  where
    ok     = filter (\k -> ((k `elem` actions) || (isLower . head $ k))
                        && not ("steps" `isInfixOf` k) 
                        && not ("vn_" `isPrefixOf` k)
                    ) observations
    idx    = T.toDType T.Int32 . toTensor 
           $ map (fromJust . flip elemIndex observations) ok

    idxI   = map (fromJust . flip elemIndex ok) 
           $ filter (\i -> ("i" `isPrefixOf` i) || (":id" `isSuffixOf` i)) ok
    mskI   = T.toDType T.Bool . toTensor $ map (`elem` idxI) [0 .. (length ok - 1)]

    frqs   = ["ugbw", "cof", "sr_f", "sr_r"] :: [[Char]]
    idxF   = map (fromJust . flip elemIndex ok) 
           $ filter (\f -> not ("delta_" `isPrefixOf` f) 
                        && (any (`isInfixOf` f) frqs
                        || (":fug" `isSuffixOf` f))
                    ) ok
    mskF   = T.toDType T.Bool . toTensor $ map (`elem` idxF) [0 .. (length ok - 1)]

    obs'   = T.indexSelect 1 idx obs
    obs''  = T.where' mskF obs' (T.log10 . T.abs $ obs')
    states = T.where' mskI obs'' (obs'' * 1.0e6)

-- | Scale reward to center
scaleRewards :: T.Tensor -> Float -> T.Tensor
scaleRewards reward factor = reward'
  where
    factor' = toTensor factor
    reward' = (reward - T.mean reward) / (T.std reward + factor')
 
-- | Obtain current performance from a gace server and write/append to log
writeEnvLog :: FilePath -> HymURL -> IO ()
writeEnvLog logPath aceUrl = do
    performance <- M.map (M.map (: [])) <$> hymPoolMap aceUrl performanceRoute 
    sizing <- M.map (M.map (: [])) <$> hymPoolMap aceUrl sizingRoute 
    let envData = M.unionWith M.union performance sizing
    fex <- doesFileExist jsonPath
    logData' <- if fex then M.unionWith (M.unionWith (++)) envData 
                                . fromJust . decodeStrict 
                                <$> BS.readFile jsonPath
                       else return envData
    BL.writeFile jsonPath (encode logData')
  where
    performanceRoute = "current_performance"
    sizingRoute      = "current_sizing"
    jsonPath         = logPath ++ "/env.json"

-- | Write an arbitrary Map to a log file for visualization
writeLog :: FilePath -> M.Map String [Float] -> IO ()
writeLog logPath logData = do
    fex <- doesFileExist logPath
    logData' <- if fex then M.unionWith (++) logData . fromJust . decodeStrict
                            <$> BS.readFile jsonPath
                       else return logData
    BL.writeFile jsonPath (encode logData')
  where
    jsonPath = logPath ++ "/log.json"
