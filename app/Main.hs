{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Lib
import qualified SAC
-- import qualified TD3
-- import qualified PPO

-- import Torch as T
-- import Criterion.Main

main :: IO ()
main = do
    putStrLn $ "Training " ++ SAC.algorithm ++ " ACiD Agent"
    acid <- SAC.train obs act url
    SAC.saveAgent acid ptPath
    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"
  where 
    host       = "localhost"
    port       = "7009"
    aceID      = "op2"
    aceBackend = "xh035"
    aceVariant = "0"
    url        = aceURL host port aceID aceBackend aceVariant
    act        = 10
    obs        = 46
    ptPath     = "./models/sac"

-- main :: IO()
-- main = defaultMain [ bench "train" $ whnfIO (SAC.train obs act url) ]
--       where 
--         host       = "localhost"
--         port       = "7009"
--         aceID      = "op2"
--         aceBackend = "xh035"
--         aceVariant = "0"
--         url        = aceURL host port aceID aceBackend aceVariant
--         act        = 10
--         obs        = 58
