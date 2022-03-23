{-# OPTIONS_GHC -Wall #-}

module Normal ( Normal (..) 
              , prob
              ) where

import qualified Torch                            as T
import qualified Torch.Distributions.Distribution as D
import qualified Torch.Distributions.Constraints  as D

-- | Creates a normal (also called Gaussian) distribution parameterized by
-- `loc` and `scale`.
-- Example:
--  
-- >>> m = Normal (asTensor ([0.0] :: [Float])) (asTensor ([1.0] :: [Float]))
-- >>> sample m []
-- Tensor Float [1] [-0.1205   ]
--
data Normal = Normal {  -- | mean of the distribution (often referred to as mu)
                       loc   :: T.Tensor
                        -- | standard deviation of the distribution (often
                        -- referred to as sigma)
                     , scale :: T.Tensor
                     } deriving Show

instance D.Distribution Normal where
  batchShape d = T.shape . loc $ d
  eventShape _ = []
  expand d s   = Normal {loc = loc', scale = scale'}
    where
      loc'     = T.expand (loc d)   True s
      scale'   = T.expand (scale d) True s
  support _    = D.real
  mean         = loc
  variance     = T.pow (2.0 :: Float) . scale
  sample d s   = do
        x <- if null s then T.randnLikeIO $ loc d
                       else T.randnIO s opts
        pure $ x * scale d + loc d
    where
      dv   = T.device $ loc d
      dt   = T.dtype  $ loc d
      opts = T.withDType dt . T.withDevice dv $ T.defaultOpts
  logProb d x  = T.log $ prob d x
  entropy d    = 0.5 * T.log (e * 2.0 * pi' * T.pow (2.0 :: Float) sigma)
    where
      dv       = T.device $ loc d
      dt       = T.dtype  $ loc d
      opts     = T.withDType dt . T.withDevice dv $ T.defaultOpts
      e        = T.exp $ T.ones [1] opts
      pi'      = T.asTensor' (pi :: Float) opts
      sigma    = scale d
  enumerateSupport = undefined

-- | PDF for given Normal Distribution
prob :: Normal -> T.Tensor -> T.Tensor
prob d x = (1.0 / (mu * T.sqrt (2.0 * pi')))
         * T.exp ((- 0.5) * T.pow (2.0 :: Float) ((x - mu) / sigma))
  where
    mu    = loc d
    sigma = scale d
    dv    = T.device mu
    dt    = T.dtype  mu
    opts  = T.withDType dt . T.withDevice dv $ T.defaultOpts
    pi'   = T.asTensor' (pi :: Float) opts
