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

import Data.Char        (isLower)
import Data.List        (isInfixOf, isPrefixOf, isSuffixOf, elemIndex)
import Data.Maybe       (fromJust)
import Data.Aeson
import Network.Wreq
import Control.Lens
import Control.Monad
import GHC.Generics
import GHC.Float        (float2Double)
import Numeric.Limits   (maxValue, minValue)
import System.Directory
import qualified Data.Map                  as M
import qualified Data.ByteString.Lazy      as BL
import qualified Data.ByteString           as BS hiding (pack)
import qualified Data.ByteString.Char8     as BS (pack)
import qualified Torch                     as T
import qualified Torch.NN                  as NN
import qualified Torch.Lens                as TL
import qualified Torch.Functional.Internal as T (where', nan_to_num)
import qualified Torch.Initializers        as T (xavierNormal)

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
data Info = Info { observations :: ![String]
                 , actions      :: ![String] 
                 } deriving (Generic, Show)
instance FromJSON Info
instance ToJSON Info

-- | Environment Step
data Step = Step { observation :: ![Float]
                 , reward      :: !Float
                 , done        :: !Bool
                 , info        :: !Info 
                 } deriving (Generic, Show)
instance FromJSON Step
instance ToJSON Step

-- | Base Route to Hym Server
type HymURL = String

-- | Possible Action Spaces
data ActionSpace = Continuous | Discrete
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
    --don  = T.reshape [-1,1] . toTensor . M.elems . M.map done        $ steps
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

-- | Process / Sanitize the Observations from GACE
processGace :: T.Tensor -> Info -> T.Tensor
processGace obs Info {..} = states
  where
    ok      = filter (\k -> ((k `elem` actions) || (isLower . head $ k))
                         && not ("steps" `isInfixOf` k) 
                         && not ("vn_" `isPrefixOf` k)
                         && not ("v_" `isPrefixOf` k)
                         && ("iss" /= k) && ("idd" /= k)
                     ) observations
    idx     = T.toDType T.Int32 . toTensor 
            $ map (fromJust . flip elemIndex observations) ok
    idxI    = map (fromJust . flip elemIndex ok) 
            $ filter (\i -> ("i_" `isInfixOf` i) || (":id" `isSuffixOf` i)) ok
    mskI    = T.toDType T.Bool . toTensor $ map (`elem` idxI) [0 .. (length ok - 1)]
    frqs    = ["ugbw", "cof", "sr_f", "sr_r"] :: [[Char]]
    idxF    = map (fromJust . flip elemIndex ok) 
            $ filter (\f -> any (`isInfixOf` f) frqs || (":fug" `isSuffixOf` f)) ok
    mskF    = T.toDType T.Bool . toTensor $ map (`elem` idxF) [0 .. (length ok - 1)]
    idxV    = map (fromJust . flip elemIndex ok) $ filter ("voff_" `isPrefixOf`) ok
    mskV    = T.toDType T.Bool . toTensor $ map (`elem` idxV) [0 .. (length ok - 1)]
    obs'    = T.indexSelect 1 idx obs
    obs''   = T.where' mskF (T.log10 . T.abs $ obs') obs' 
    obs'''  = T.where' mskI (obs'' * 1.0e6) obs''
    obs'''' = T.where' mskV (obs''' * 1.0e3) obs'''
    states  = nanToNum'' obs''''

-- | Scale reward to center
scaleRewards :: T.Tensor -> Float -> T.Tensor
scaleRewards reward factor = (reward - T.mean reward) / (T.std reward + factor')
  where
    factor' = toTensor factor
 
------------------------------------------------------------------------------
-- Data Logging
------------------------------------------------------------------------------

-- | Create a log directory
createLogDir :: FilePath -> IO ()
createLogDir = createDirectoryIfMissing True

-- | Setup the Logging Direcotry for SHACE
setupLogging :: FilePath -> IO ()
setupLogging remoteDir = do
    localLoss   <- (++ "/log/loss.csv")   <$> getCurrentDirectory
    localReward <- (++ "/log/reward.csv") <$> getCurrentDirectory

    BS.writeFile localLoss   "Iteration,Model,Loss\n"
    BS.writeFile localReward "Iteration,Env,Reward\n"

    createLogDir remoteDir

    doesFileExist remoteLoss   >>= flip when (removeFile remoteLoss)
    doesFileExist remoteReward >>= flip when (removeFile remoteReward)

    createFileLink localLoss   remoteLoss
    createFileLink localReward remoteReward
  where
    remoteLoss   = remoteDir ++ "/loss.csv"
    remoteReward = remoteDir ++ "/reward.csv"

-- | Get SHACE Logging path to a given URL
remoteLogPath :: HymURL -> IO FilePath
remoteLogPath url = (++ "/model") <$> shaceLogPath url

-- | Append a line to the given loss log file (w/o) episode
writeLoss :: Int -> String -> Float -> IO ()
writeLoss iteration model loss = BS.appendFile path line
  where
    path = "./log/loss.csv"
    line = BS.pack $ show iteration ++ "," ++ model ++ "," ++ show loss ++ "\n"

-- | Append a line to the given reward log file (w/o) episode
writeReward :: Int -> T.Tensor -> IO ()
writeReward iteration rewards = forM_ (zip3 (repeat iteration) env reward)
                                      (\(i,e,r) -> BS.appendFile path 
                                               <$> BS.pack 
                                                $  show i ++ "," ++ show e 
                                                          ++ "," ++ show r 
                                                          ++ "\n")
  where
    path   = "./log/reward.csv"
    reward = T.asValue (T.squeezeAll rewards) :: [Float]
    env    = [ 0 .. (length reward - 1) ]

-- | Obtain current performance from a gace server and write/append to log
writeEnvLog :: FilePath -> HymURL -> IO ()
writeEnvLog path aceUrl = do
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
    jsonPath         = path ++ "/env.json"

-- | Write an arbitrary Map to a log file for visualization
writeLog :: FilePath -> M.Map String [Float] -> IO ()
writeLog path logData = do
    fex <- doesFileExist path
    logData' <- if fex then M.unionWith (++) logData . fromJust . decodeStrict
                            <$> BS.readFile jsonPath
                       else return logData
    BL.writeFile jsonPath (encode logData')
  where
    jsonPath = path ++ "/log.json"

------------------------------------------------------------------------------
-- Data Visualization
------------------------------------------------------------------------------
