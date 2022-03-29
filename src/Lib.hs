{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Utility and Helper functions for EDELWACE
module Lib where

import Data.Char             (isLower)
import Data.List             (isInfixOf, isPrefixOf, isSuffixOf, elemIndex)
import Data.Maybe            (fromJust)
import Data.Aeson
import Network.Wreq
import Control.Lens
import Control.Monad
import GHC.Generics
import GHC.Float             (float2Double)
import Numeric.Limits        (maxValue, minValue)
import Data.Time.Clock.POSIX (getPOSIXTime)
-- import System.Directory
import qualified Data.Map                  as M
import qualified Data.ByteString.Lazy      as BL
import qualified Data.ByteString           as BS hiding (pack)
-- import qualified Data.ByteString.Char8     as BS (pack)
import qualified Torch                     as T
import qualified Torch.NN                  as NN
import qualified Torch.Lens                as TL
import qualified Torch.Functional.Internal as T         (where', nan_to_num)
import qualified Torch.Initializers        as T         (xavierNormal)
import qualified MLFlow                    as MLF
import qualified MLFlow.DataStructures     as MLF

------------------------------------------------------------------------------
-- Convenience / Syntactic Sugar
------------------------------------------------------------------------------

-- | Swaps the arguments of HaskTorch's foldLoop around
foldLoop' :: Int -> (a -> Int -> IO a) -> a -> IO a
foldLoop' i f m = T.foldLoop m i f

-- | Because snake_case sucks
nanToNum :: Float -> Float -> Float -> T.Tensor -> T.Tensor
nanToNum nan' posinf' neginf' self = T.nan_to_num self nan posinf neginf
  where
    nan    = float2Double nan'
    posinf = float2Double posinf'
    neginf = float2Double neginf'

-- | Default limits for `nanToNum`
nanToNum' :: T.Tensor -> T.Tensor
nanToNum' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = float2Double (maxValue :: Float)
    neginf = float2Double (minValue :: Float)

-- | Default limits for `nanToNum` (0.0)
nanToNum'' :: T.Tensor -> T.Tensor
nanToNum'' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = 0.0 :: Double
    neginf = 0.0 :: Double

-- | GPU Tensor filled with Float value
fullLike' :: T.Tensor -> Float -> T.Tensor
fullLike' self num = T.onesLike self * toTensor num

-- | Select index with [Int] from GPU tensor
indexSelect'' :: Int -> [Int] -> T.Tensor -> T.Tensor
indexSelect'' dim idx ten = ten'
  where
    opts = T.withDType T.Int32 . T.withDevice (T.device ten) $ T.defaultOpts
    idx' = T.asTensor' idx opts
    ten' = T.indexSelect dim idx' ten

------------------------------------------------------------------------------
-- Neural Networks
------------------------------------------------------------------------------

-- | Calculate weight Limits based on Layer Dimensions
weightLimit :: T.Linear -> Float
weightLimit layer = fanIn ** (- 0.5)
  where
    fanIn = realToFrac . head . T.shape . T.toDependent . NN.weight $ layer

-- | Initialize Weights of Linear Layer
weightInit :: Float -> T.Linear -> IO T.Linear
weightInit limit layer = do
    weight' <- T.xavierNormal limit dims >>= T.makeIndependent
    pure T.Linear { NN.weight = weight', NN.bias = bias' }
  where
    dims  = T.shape . T.toDependent . NN.weight $ layer
    bias' = NN.bias layer

-- | Initialize weights based on Fan In
weightInit' :: T.Linear -> IO T.Linear
weightInit' layer = weightInit limit layer
  where
    limit = weightLimit layer

-- | Softly update parameters from Online Net to Target Net
softUpdate :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
softUpdate τ t o = (t * (o' - τ)) + (o * τ)
  where
    o' = T.onesLike τ

-- | Softly copy parameters from Online Net to Target Net
softSync :: NN.Parameterized f => T.Tensor -> f -> f -> IO f
softSync τ target online =  NN.replaceParameters target 
                        <$> mapM T.makeIndependent tUpdate 
  where
    tParams = fmap T.toDependent . NN.flattenParameters $ target
    oParams = fmap T.toDependent . NN.flattenParameters $ online
    tUpdate = zipWith (softUpdate τ) tParams oParams

-- | Hard Copy of Parameter from one net to the other
copySync :: NN.Parameterized f => f -> f -> f
copySync target =  NN.replaceParameters target . NN.flattenParameters

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

-- | Convert an Array to a Tensor
toIntTensor :: T.TensorLike a => a -> T.Tensor
toIntTensor t = T.asTensor' t opts
  where
    opts = T.withDType T.Int32 . T.withDevice gpu $ T.defaultOpts

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

------------------------------------------------------------------------------
-- Statistics
------------------------------------------------------------------------------

-- | Generate a Tensor of random Integers
randomInts :: Int -> Int -> Int -> IO T.Tensor
randomInts lo hi num = T.randintIO lo hi [num] opts 
  where
    opts = T.withDType T.Int64 . T.withDevice gpu $ T.defaultOpts

-- | Generate Normally Distributed Random values given dimensions
normal' :: [Int] -> IO T.Tensor
normal' dims = T.randnIO dims opts
  where
    opts = T.withDType T.Float . T.withDevice gpu $ T.defaultOpts

-- | Generate Normally Distributed Random values given μs and σs
normal :: T.Tensor -> T.Tensor -> IO T.Tensor
normal μ σ = toFloatGPU <$> T.normalIO μ σ

------------------------------------------------------------------------------
-- Hym Server Interaction and Environment
------------------------------------------------------------------------------

-- | Info object gotten form stepping
data Info = Info { observations :: ![String]    -- ^ Observation Keys
                 , actions      :: ![String]    -- ^ Action Keys
                 } deriving (Generic, Show)

instance FromJSON Info
instance ToJSON Info

-- | Single Environment Step
data Step = Step { observation :: ![Float]  -- ^ Observation Vector
                 , reward      :: !Float    -- ^ Reward Scalar
                 , done        :: !Bool     -- ^ Terminal Indicator
                 , info        :: !Info     -- ^ Info
                 } deriving (Generic, Show)

instance FromJSON Step
instance ToJSON Step

-- | Base Route to Hym Server
type HymURL = String

-- | Possible Action Spaces
data ActionSpace = Continuous -- ^ Continuous Action Space
                 | Discrete   -- ^ Discrete Action Space
    deriving (Show, Eq)

-- | Convert a Map to a Tensor where Pool index is a dimension
mapToTensor :: M.Map Int [Float] -> T.Tensor
mapToTensor = toTensor . M.elems

-- | Convert Tensor to Map (Continuous action spaces)
tensorToMap :: T.Tensor -> M.Map Int [Float]
tensorToMap = M.fromList . zip [0 .. ] . T.asValue

-- | Convert Tensor to Map (Discrete action spaces)
tensorToMap' :: T.Tensor -> M.Map Int Int
tensorToMap' = M.fromList . zip [0 .. ] . T.asValue

-- | Convert the Pooled Step Map to a Tuple
stepsToTuple :: M.Map Int Step -> (T.Tensor, T.Tensor, T.Tensor, [Info])
stepsToTuple steps = (obs, rew, don, inf)
  where
    opts = T.withDType T.Bool . T.withDevice gpu $ T.defaultOpts
    obs  =                    toTensor . M.elems . M.map observation      $ steps
    rew  = T.reshape [-1,1] . toTensor . M.elems . M.map reward           $ steps
    don  = T.reshape [-1,1] . (`T.asTensor'` opts) . M.elems . M.map done $ steps
    inf  =                               M.elems . M.map info             $ steps

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

-- | Convert a JSON Response from an ACE Server to a Float-List
hymPoolList :: HymURL -> String -> IO (M.Map Int [Float])
hymPoolList url route = fromJust . decodeStrict <$> hymGet url route

-- | Convert a JSON Response from an ACE Server to a String-List
hymPoolList' :: HymURL -> String -> IO (M.Map Int [String])
hymPoolList' url route = fromJust . decodeStrict <$> hymGet url route

-- | Reset Pooled Environments on a Hym server
hymPoolReset :: HymURL -> IO (M.Map Int [Float])
hymPoolReset = flip hymPoolList "reset"

-- | Get Random Actions from all Pooled Environments
hymPoolRandomAction :: HymURL -> IO (M.Map Int [Float])
hymPoolRandomAction = flip hymPoolList "random_action"

-- | Perform Random Actions in all Pooled Environments
hymPoolRandomStep :: HymURL -> IO (M.Map Int Step)
hymPoolRandomStep url = fromJust . decodeStrict <$> hymGet url "random_step"

-- | Take Steps in All Environments (Continuous)
hymPoolStep :: HymURL -> M.Map Int [Float] -> IO (M.Map Int Step)
hymPoolStep url action = fromJust . decodeStrict
                      <$> hymPost url "step" (toJSON . M.mapKeys show $ action)

-- | Take Steps in All Environments (Discrete)
hymPoolStep' :: HymURL -> M.Map Int Int -> IO (M.Map Int Step)
hymPoolStep' url action = fromJust . decodeStrict
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

-- | Get the SHACE logging path as a dict
shaceLogPath' :: HymURL -> IO (M.Map String String)
shaceLogPath' url = fromJust . decodeStrict <$> hymGet url "log_path"

-- | Get the SHACE logging path
shaceLogPath :: HymURL -> IO String
shaceLogPath url = fromJust . M.lookup "path" <$> shaceLogPath' url

-- | Reset a Vectorized Environment Pool
resetPool :: HymURL -> IO T.Tensor
resetPool url = mapToTensor <$> hymPoolReset url

-- | Reset selected Environments from Pool
resetPool' :: HymURL -> T.Tensor -> IO T.Tensor
resetPool' url mask = mapToTensor . fromJust . decodeStrict 
                   <$> hymPost url "reset" payload
  where
    mask' = T.asValue (T.squeezeAll mask) :: [Bool]
    dict = M.fromList [("done_mask", mask')] :: M.Map String [Bool]
    payload = toJSON dict

-- | Shorthand for getting keys of pooled same envs
actKeysPool :: HymURL -> IO [String]
actKeysPool url = fromJust . M.lookup 0 <$> acePoolActKeys url 

-- | Shorthand for getting keys of pooled same envs
obsKeysPool :: HymURL -> IO [String]
obsKeysPool url = fromJust . M.lookup 0 <$> acePoolObsKeys url 

-- | Number of Environments in Pool
numEnvsPool :: HymURL -> IO Int
numEnvsPool url = getNum . fromJust . decodeStrict <$> hymGet url route
  where
    route = "num_envs"
    getNum :: M.Map String Int -> Int
    getNum = fromJust . M.lookup "num"

-- | Get Info without stepping
infoPool :: HymURL -> IO Info
infoPool url = do
    obs <- obsKeysPool url
    act <- actKeysPool url
    return (Info obs act)

-- | Step in a Control Environment
stepPool :: HymURL -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor, [Info])
stepPool url action = stepsToTuple <$> hymPoolStep url (tensorToMap action)

-- | Step in a Discrete Environment
stepPool' :: HymURL -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor, [Info])
stepPool' url action = stepsToTuple 
                    <$> (hymPoolStep' url . tensorToMap' . T.squeezeAll $ action)

-- | Take a random Step an Environment
randomStepPool :: HymURL -> IO (T.Tensor, T.Tensor, T.Tensor, [Info])
randomStepPool url = stepsToTuple <$> hymPoolRandomStep url

-- | Get a set of random actions from the current environment
randomActionPool :: HymURL -> IO T.Tensor
randomActionPool url = mapToTensor <$> hymPoolRandomAction url

-- | Optimizer moments at given prefix
saveOptim :: T.Adam -> FilePath -> IO ()
saveOptim optim prefix = do
    T.save (T.m1 optim) (prefix ++ "M1.pt")
    T.save (T.m2 optim) (prefix ++ "M2.pt")

-- | Load Optimizer State
loadOptim :: Int -> Float -> Float -> FilePath -> IO T.Adam
loadOptim iter β1 β2 prefix = do
    m1' <- T.load (prefix ++ "M1.pt")
    m2' <- T.load (prefix ++ "M2.pt")
    pure $ T.Adam β1 β2 m1' m2' iter

------------------------------------------------------------------------------
-- Data Processing
------------------------------------------------------------------------------

-- | Create Boolean Mask Tensor from list of indices.
boolMask :: Int -> [Int] -> T.Tensor
boolMask len idx = mask
  where
    mask = T.toDType T.Bool . toTensor $ map (`elem` idx) [0 .. (len - 1)]

-- | Process / Sanitize the Observations from GACE
processGace :: T.Tensor -> Info -> T.Tensor
processGace obs Info {..} = states
  where
    ok      = filter (\k -> ( (k `elem` actions) 
                           || (isLower . head $ k)
                           || (k == "A") )
                         && not ("steps" `isInfixOf` k) 
                         && not ("vn_" `isPrefixOf` k)
                         && not ("v_" `isPrefixOf` k)
                         && ("iss" /= k) && ("idd" /= k)
                     ) observations
    idx     = T.toDType T.Int32 . toTensor 
            $ map (fromJust . flip elemIndex observations) ok
    idxI    = map (fromJust . flip elemIndex ok) 
            $ filter (\i -> ("i_" `isInfixOf` i) || (":id" `isSuffixOf` i)) ok
    mskI    = boolMask (length ok) idxI
    frqs    = ["ugbw", "cof", "sr_f", "sr_r"] :: [[Char]]
    idxF    = map (fromJust . flip elemIndex ok) 
            $ filter (\f -> any (`isInfixOf` f) frqs 
                             || (":fug" `isSuffixOf` f)) ok
    mskF    = boolMask (length ok) idxF
    idxV    = map (fromJust . flip elemIndex ok) 
            $ filter ("voff_" `isPrefixOf`) ok
    mskV    = boolMask (length ok) idxV
    idxA    = [fromJust $ elemIndex "A" ok]
    mskA    = boolMask (length ok) idxA
    obs1    = T.indexSelect 1 idx obs
    obs2    = T.where' mskF (T.log10 . T.abs $ obs1) obs1 
    obs3    = T.where' mskI (obs2 * 1.0e6)  obs2
    obs4    = T.where' mskV (obs3 * 1.0e3)  obs3
    obs5    = T.where' mskA (obs4 * 1.0e10) obs4
    states  = nanToNum'' obs5

-- | Scale reward to center
scaleRewards :: T.Tensor -> Float -> T.Tensor
scaleRewards reward factor = (reward - T.mean reward) / (T.std reward + factor')
  where
    factor' = toTensor factor
 
------------------------------------------------------------------------------
-- Data Logging / Visualization
------------------------------------------------------------------------------

-- | Sanatize JSON for MLFlow: Names may only contain alphanumerics,
-- underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/).
sanatizeJSON :: Char -> Char
sanatizeJSON ':' = '_'  -- Replace Colons
sanatizeJSON ';' = '_'  -- Replace Semicolons
sanatizeJSON ',' = '_'  -- Replace Commas
sanatizeJSON  c  =  c   -- Leave as is

-- | Data Logging to MLFlow Trackign Server
data Tracker = Tracker { uri            :: MLF.TrackingURI        -- ^ Tracking Server URI
                       , experimentId   :: MLF.ExperimentID       -- ^ Experiment ID
                       , experimentName :: String                 -- ^ Experiment Name
                       , runIds         :: M.Map String MLF.RunID -- ^ Run IDs
                       } deriving (Show)

-- | Retrieve a run ID
runId :: Tracker -> String -> MLF.RunID
runId Tracker{..} id' = fromJust $ M.lookup id' runIds

-- | Make new Tracker given a Tracking Server URI
mkTracker :: MLF.TrackingURI -> String -> IO Tracker
mkTracker uri' expName = do
    suffix <- (round . (* 1000) <$> getPOSIXTime :: IO Int)
    let expName' = expName ++ "_" ++ show suffix
    expId' <- MLF.createExperiment uri' expName'
    pure (Tracker uri' expId' expName' M.empty)

-- | Make new Tracker given a Hostname and Port
mkTracker' :: String -> Int -> String -> IO Tracker
mkTracker' host port = mkTracker (MLF.trackingURI' host port)

---- | Create a new Experiment with rng suffix
newExperiment :: Tracker -> String -> IO Tracker
newExperiment Tracker{..} expName = do
    suffix <- (round . (* 1000) <$> getPOSIXTime :: IO Int)
    let expName' = expName ++ "_" ++ show suffix
    expId' <- MLF.createExperiment uri expName'
    pure (Tracker uri expId' expName' M.empty)

---- | Create a new Experiment
newExperiment' :: Tracker -> String -> IO Tracker
newExperiment' Tracker{..} expName = do
    expId' <- MLF.createExperiment uri expName
    pure (Tracker uri expId' expName M.empty)

---- | Create a new run with a set of given paramters
newRuns :: Tracker -> [String] -> [MLF.Param] -> IO Tracker
newRuns Tracker{..} ids params' = do
    unless (M.null runIds) do
        forM_ (M.elems runIds) (MLF.endRun uri)
        putStrLn "Ended runs before starting new ones."
    runIds' <- replicateM (length ids) 
                 (MLF.runId . MLF.runInfo <$> MLF.createRun uri experimentId [])
    forM_ (zip runIds' params') (\(rid, p') -> MLF.logBatch uri rid [] [p'] [])
    let runs = M.fromList $ zip ids runIds'
    pure (Tracker uri experimentId experimentName runs)

---- | New run with algorithm id and #envs as log params
newRuns' :: Int -> Tracker -> IO Tracker
newRuns' numEnvs tracker = newRuns tracker ids params'
  where
    ids     = map (("env_" ++) . show) [0 .. (numEnvs - 1)] 
           ++ ["reward", "loss"]
    params' = map (MLF.Param "id" . show) [0 .. (numEnvs - 1)] 
           ++ [MLF.Param "id" "rewards", MLF.Param "id" "losses"]

---- | End a run
endRun :: String -> Tracker -> IO Tracker
endRun id' tracker@Tracker{..} = do
    _ <- MLF.endRun uri (runId tracker id')
    pure (Tracker uri experimentId experimentName runIds')
  where 
    runIds' = M.delete id' runIds

---- | Write Loss to Tracking Server
trackLoss :: Tracker -> Int -> String -> Float -> IO (Response BL.ByteString)
trackLoss tracker@Tracker{..} epoch ident loss = 
    MLF.logMetric uri runId' ident loss epoch
  where
    runId' = runId tracker "loss" 

---- | Write Reward to Tracking Server
trackReward :: Tracker -> Int -> T.Tensor -> IO ()
trackReward tracker@Tracker{..} step reward = do
        let rewId = runId tracker "reward"
            rAvg  = T.asValue (T.mean reward) :: Float
            rSum  = T.asValue (T.sumAll reward) :: Float
            rMin  = T.asValue (T.min reward) :: Float
            rMax  = T.asValue (T.max reward) :: Float
        _ <- MLF.logMetric uri rewId "sum" rSum step
        _ <- MLF.logMetric uri rewId "avg" rAvg step
        _ <- MLF.logMetric uri rewId "max" rMax step
        _ <- MLF.logMetric uri rewId "min" rMin step
        forM_ (zip envIds rewards) 
            (\(envId, rewardValue) -> 
                let runId' = runId tracker envId
                 in MLF.logMetric uri runId' "reward" rewardValue step)
  where
    rewards = T.asValue (T.squeezeAll reward) :: [Float]
    envIds  = [ "env_" ++ show e | e <- [0 .. (length rewards - 1) ]]

---- | Filter Performance of all envs
filterPerformance :: M.Map Int (M.Map String Float) -> [String] 
                  -> M.Map Int (M.Map String Float)
filterPerformance performance keys = M.map keyFilter performance
  where
    keyFilter m = M.fromList $ [ (map sanatizeJSON k, fromJust $ M.lookup k m ) 
                               | k <- M.keys m, k `elem` keys ]
    --keyFilter = M.filterWithKey (\k _ -> k `elem` keys)

---- | Write Current state of the Environment to Trackign Server
trackEnvState :: Tracker -> HymURL -> Int -> IO ()
trackEnvState tracker@Tracker{..} url step = do
    performance'    <- hymPoolMap url performanceRoute 
    sizing          <- hymPoolMap url sizingRoute 
    actions         <- filterPerformance performance' . head . M.elems 
                    <$> hymPoolList' url actionRoute
    target'         <- hymPoolMap url targetRoute
    let targetKeys  = M.keys . head . M.elems $ target'
        performance = filterPerformance performance' targetKeys
        target      = M.map (M.mapKeys (++ "_target")) target'

    forM_ (M.keys target)
          (\id' -> 
              let envId  = "env_" ++ show id'
                  state  = M.unions 
                         $ map (fromJust . M.lookup id')
                               [target, performance, sizing, actions]
                  runId' = runId tracker envId
               in MLF.logBatch' uri runId' step state M.empty)
  where
    performanceRoute = "current_performance"
    sizingRoute      = "current_sizing"
    targetRoute      = "target"
    actionRoute      = "action_keys"
