{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module SAC ( algorithm
           , train
           , saveModels
           , loadActor
           , loadCritic
           , loadValue
           , actorForward
           , criticForward
           , valueForward
           , act
           , evaluate
           , ActorNet (..)
           , ActorNetSpec (..)
           , CriticNet (..)
           , CriticNetSpec (..)
           , ValueNet (..)
           , ValueNetSpec (..)
           ) where

import Lib

import Data.List (elemIndex)
import Data.Maybe (fromJust)
import GHC.Generics
import qualified Data.Random as RNG
import qualified Torch as T
import qualified Torch.NN as NN

-- | Algorithm ID
algorithm :: String
algorithm = "sac"

-- | Algorithm Settings
γ    :: Float
γ    = 0.99
τ    :: Float
τ    = 1.0e-2
μλ   :: Float
μλ   = 1.0e-3
σλ   :: Float
σλ   = 1.0e-3
zλ   :: Float
zλ   = 0.0
ε    :: Float
ε    = 1.0e-6
α    :: Float
α    = 3.0e-3   -- 3.0e-4
β1   :: Float
β1   = 0.75     -- 0.9
β2   :: Float
β2   = 0.98     -- 0.99
σMin :: Float
σMin = -2.0
σMax :: Float
σMax = 20.0
                 
-- | Neural Network Specification
newtype ValueNetSpec = ValueNetSpec { vObsDim :: Int }
    deriving (Show, Eq)

data CriticNetSpec = CriticNetSpec { cObsDim :: Int, cActDim :: Int }
    deriving (Show, Eq)

data ActorNetSpec = ActorNetSpec { aObsDim :: Int, aActDim :: Int }
    deriving (Show, Eq)

-- | Network Architecture
data ValueNet = ValueNet { vLayer0 :: T.Linear
                         , vLayer1 :: T.Linear
                         , vLayer2 :: T.Linear
                         , vLayer3 :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

data CriticNet = CriticNet { cLayer0 :: T.Linear
                           , cLayer1 :: T.Linear
                           , cLayer2 :: T.Linear
                           , cLayer3 :: T.Linear }
    deriving (Generic, Show, T.Parameterized)

data ActorNet = ActorNet { aLayer0 :: T.Linear
                         , aLayer1 :: T.Linear
                         , aLayer2 :: T.Linear
                         , aLayerμ :: T.Linear
                         , aLayerσ :: T.Linear }
    deriving (Generic, Show, T.Parameterized)


-- | Neural Network Weight initialization
instance T.Randomizable ValueNetSpec ValueNet where
    sample ValueNetSpec {..} = ValueNet <$> T.sample (T.LinearSpec vObsDim 256)
                                        <*> T.sample (T.LinearSpec 256     128)
                                        <*> T.sample (T.LinearSpec 128     64)
                                        <*> T.sample (T.LinearSpec 64      1)

instance T.Randomizable CriticNetSpec CriticNet where
    sample CriticNetSpec {..} = CriticNet <$> T.sample (T.LinearSpec (cObsDim + cActDim) 256)
                                          <*> T.sample (T.LinearSpec 256                 128)
                                          <*> T.sample (T.LinearSpec 128                 64)
                                          <*> T.sample (T.LinearSpec 64                  1)

instance T.Randomizable ActorNetSpec ActorNet where
    sample ActorNetSpec {..} = ActorNet <$> T.sample (T.LinearSpec aObsDim 256)
                                        <*> T.sample (T.LinearSpec 256     128)
                                        <*> T.sample (T.LinearSpec 128     64)
                                        <*> T.sample (T.LinearSpec 64      aActDim)
                                        <*> T.sample (T.LinearSpec 64      aActDim)

type SACModel = (ActorNet, CriticNet, ValueNet, ValueNet)

-- | Neural Network Forward Pass
valueForward :: ValueNet -> T.Tensor -> T.Tensor
valueForward ValueNet {..} = T.linear vLayer3 . T.relu
                           . T.linear vLayer2 . T.relu
                           . T.linear vLayer1 . T.relu
                           . T.linear vLayer0

criticForward :: CriticNet -> T.Tensor -> T.Tensor -> T.Tensor
criticForward CriticNet {..} o a = T.linear cLayer3 . T.relu
                                 . T.linear cLayer2 . T.relu
                                 . T.linear cLayer1 . T.relu
                                 . T.linear cLayer0 $ input
  where
    input = T.cat (T.Dim $ -1) [o,a]

actorForward :: ActorNet -> T.Tensor -> (T.Tensor, T.Tensor)
actorForward ActorNet {..} obs = (μ, σ)
  where
    x = T.linear aLayer2 . T.relu
      . T.linear aLayer1 . T.relu
      . T.linear aLayer0 $ obs
    μ = T.linear aLayerμ x
    σ = T.clamp σMin σMax . T.linear aLayerσ $ x

--evaluate :: ActorNet -> T.Tensor -> IO (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)
evaluate :: RNG.MonadRandom m => ActorNet -> T.Tensor 
         -> m (T.Tensor, T.Tensor, T.Tensor, T.Tensor, T.Tensor)
evaluate actor state = do
    let (μ', σ') = actorForward actor state
        σ       = T.asValue . T.exp $ σ' :: [[Float]]
        μ       = T.asValue μ' :: [[Float]]
        n       = zipWith (zipWith RNG.Normal) μ σ :: [[RNG.Normal Float]]
    z' <- mapM (mapM RNG.sample) n
    let z       = toTensor z'
        a       = T.tanh z
        ε'      = T.log $ toScalar 1.0 - T.pow (2.0 :: Float) a + toScalar ε
        l       = toTensor (zipWith (zipWith RNG.logPdf) n z')
        p       = T.sumDim (T.Dim $ -1) T.KeepDim dataType (l - ε')
    return (a, p, z, μ', σ')

act :: RNG.MonadRandom m => ActorNet -> T.Tensor -> m T.Tensor
act actor state =  T.tanh . toTensor <$> mapM (mapM RNG.sample) n
  where
    (μ',σ') = actorForward actor state
    σ       = T.asValue . T.exp $ σ' :: [[Float]]
    μ       = T.asValue μ' :: [[Float]]
    n       = zipWith (zipWith RNG.Normal) μ σ :: [[RNG.Normal Float]]

-- act :: ActorNet -> T.Tensor -> IO T.Tensor
-- act actor state = do 
--     let (μ',σ') = actorForward actor state
--         σ       = T.asValue . T.exp $ σ' :: [[Float]]
--         μ       = T.asValue μ' :: [[Float]]
--         n       = zipWith (zipWith RNG.Normal) μ σ :: [[RNG.Normal Float]]
--     s <- mapM (mapM RNG.sample) n
--     let a = T.tanh . toTensor $ s
--     return a

saveModels :: ActorNet -> CriticNet -> ValueNet -> ValueNet-> String -> IO ()
saveModels ao co vo vt p = head $ zipWith T.saveParams [ao', co', vo', vt'] 
                                                       [pao, pco, pvo, pvt]
  where
    ao' = T.toDependent <$> T.flattenParameters ao
    co' = T.toDependent <$> T.flattenParameters co
    vo' = T.toDependent <$> T.flattenParameters vo
    vt' = T.toDependent <$> T.flattenParameters vt
    pao = p ++ "/actorO.pt"
    pco = p ++ "/criticO.pt"
    pvo = p ++ "/valueO.pt"
    pvt = p ++ "/valueT.pt"

loadActor :: String -> Int -> Int -> IO ActorNet
loadActor fp numObs numAct = T.sample (ActorNetSpec numObs numAct) 
                           >>= flip T.loadParams fp

loadCritic :: String -> Int -> Int -> IO CriticNet
loadCritic fp numObs numAct = T.sample (CriticNetSpec numObs numAct) 
                            >>= flip T.loadParams fp

loadValue :: String -> Int -> IO ValueNet
loadValue fp numObs = T.sample (ValueNetSpec numObs) 
                    >>= flip T.loadParams fp

softUpdate :: T.Tensor -> T.Tensor -> T.Tensor
softUpdate t o = (t * (o' - τ')) + (o * τ')
  where
    τ' = toTensor τ
    o' = T.onesLike τ'

softSync :: ValueNet -> ValueNet -> IO ValueNet
softSync target online =  NN.replaceParameters target 
                        <$> mapM T.makeIndependent tUpdate 
  where
    tParams = fmap T.toDependent . NN.flattenParameters $ target
    oParams = fmap T.toDependent . NN.flattenParameters $ online
    tUpdate = zipWith softUpdate tParams oParams

copySync :: ValueNet -> ValueNet -> ValueNet
copySync target =  NN.replaceParameters target . NN.flattenParameters

cCriterion :: T.Tensor -> T.Tensor -> T.Tensor
cCriterion = T.mseLoss

vCriterion :: T.Tensor -> T.Tensor -> T.Tensor
vCriterion = T.mseLoss

softQupdate :: (T.Optimizer o) => SACModel -> (o, o, o) -> ReplayBuffer 
            -> IO (SACModel, (o, o, o))
softQupdate nets opts ReplayBuffer {..} = do
    let vExpected = T.squeezeAll $ valueForward vNet states
        cExpected = T.squeezeAll $ criticForward cNet states actions

    (actions', logProbs', z, μ, σ) <- evaluate aNet states
    let logProbs = T.squeezeAll logProbs'

    let vTarget = T.squeezeAll $ valueForward vTgt states'
        dones'  = 1.0 - dones
        cNext   = rewards + dones' * γ' * vTarget
        cLoss   = T.squeezeAll $ cCriterion cExpected cNext

    let cExpNxt = T.squeezeAll $ criticForward cNet states actions'
        vNext   = cExpNxt - logProbs
        vLoss   = T.squeezeAll $ vCriterion vExpected vNext

    let lTarget = cExpNxt - vExpected
        lLoss   = T.mean $ logProbs * (logProbs - lTarget)
        μLoss   = (* μλ') . T.mean $ T.pow (2.0 :: Float) μ
        σLoss   = (* σλ') . T.mean $ T.pow (2.0 :: Float) σ
        zLoss   = (* zλ') . T.mean . T.sumDim (T.Dim 1) T.RemoveDim dataType 
                                   $ T.pow (2.0 :: Float) z
        aLoss   = T.squeezeAll $ lLoss + μLoss + σLoss + zLoss

    (aNet', aOpt') <- T.runStep aNet aOpt aLoss α'
    (cNet', cOpt') <- T.runStep cNet cOpt cLoss α'
    (vNet', vOpt') <- T.runStep vNet vOpt vLoss α'

    vTgt' <- softSync vTgt vNet'

    let nets' = (aNet', cNet', vNet', vTgt')
        opts' = (aOpt', cOpt', vOpt')

    return (nets', opts')
  where
    α'  = toScalar α
    γ'  = toScalar γ
    μλ' = toScalar μλ
    σλ' = toScalar σλ
    zλ' = toScalar zλ
    (aNet,cNet,vNet,vTgt) = nets
    (aOpt,cOpt,vOpt)      = opts

postProcess :: T.Tensor -> [String] -> T.Tensor
postProcess observations keys = observations'
  where
    opts = T.withDType T.Int32 . T.withDevice computingDevice $ T.defaultOpts
    idx' = map (fromJust . (`elemIndex` keys)) . filter (notElem ':') $ keys
    idx  = T.asTensor' idx' opts
    observations' = T.indexSelect 1 idx observations

train :: Int -> Int -> ACEURL -> IO SACModel
train obsDim actDim aceUrl = do
    actorOnline  <- toFloatGPU <$> T.sample (ActorNetSpec obsDim actDim)
    criticOnline <- toFloatGPU <$> T.sample (CriticNetSpec obsDim actDim)
    valueOnline  <- toFloatGPU <$> T.sample (ValueNetSpec obsDim)
    valueTarget' <- toFloatGPU <$> T.sample (ValueNetSpec obsDim)

    let valueTarget = copySync valueTarget' valueOnline

    let actorOptim  = T.mkAdam 0 β1 β2 (NN.flattenParameters actorOnline)
        criticOptim = T.mkAdam 0 β1 β2 (NN.flattenParameters criticOnline)
        valueOptim  = T.mkAdam 0 β1 β2 (NN.flattenParameters valueOnline)

    let models = (actorOnline, criticOnline, valueOnline, valueTarget)
        optims = (actorOptim, criticOptim, valueOptim)

    (nets', _) <- T.foldLoop (models, optims) numEpisodes $ 
        \(nets, opts) episode -> do
            obs_ <- toFloatGPU <$> resetPool aceUrl
            keys <- obsKeysPool aceUrl
            let obs = postProcess obs_ keys
                buffer = emptyBuffer
            (nets', opts', _, _) <- T.foldLoop (nets, opts, obs, buffer) numSteps $
                \(n, op, ob, buf) stp -> do
                    let (actOnline, _, _, _) = n
                    acts <- act actOnline ob
                    (obs'',rews,dons,_) <- stepPool aceUrl acts
                    let obs' = postProcess obs'' keys
                        newBuf = ReplayBuffer ob acts rews obs' dons
                        buffer' = if lengthBuffer buf > 0 
                                     then pushBuffer buf newBuf
                                     else newBuf

                    let options = T.withDType T.Int32 
                                . T.withDevice computingDevice 
                                $ T.defaultOpts
                    bufferSample <- sampleBuffer buffer'
                                 <$> T.randintIO 0 (lengthBuffer buffer') 
                                                 [batchSize] options

                    (nets',opts') <- softQupdate n op bufferSample

                    let r = T.asValue . T.mean $ rews :: Float

                    putStrLn $ "Step: " ++ show stp ++ " | Average Reward : "
                                        ++ show r

                    return (nets', opts', obs', buffer')

            putStrLn $ "Episode: " ++ show episode ++ " | Done."
            return (nets', opts')
       
    let (actorOnl, criticOnl, valueOnl, valueTgt) = nets'

    saveModels actorOnl criticOnl valueOnl valueTgt ptPath
    return nets'
    where 
        numEpisodes = 666 :: Int
        numSteps    = 100 :: Int
        -- earlyStop   = -50 :: Int
        batchSize   = 100 :: Int
        -- bufferSize  = round 1.0e7 :: Int
        ptPath      = "./models/sac"
        -- ckptFile    = "./models/sac/model.ckpt"

