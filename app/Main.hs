{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

-- import Torch as T
-- import Torch.NN as NN

import Lib
import qualified SAC
-- import qualified TD3
-- import qualified PPO

main :: IO ()
main = do
    putStrLn $ "Running " ++ SAC.algorithm
    _ <- SAC.train obs act url
    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"
      where 
        host       = "localhost"
        port       = "7009"
        aceID      = "op2"
        aceBackend = "xh035"
        aceVariant = "0"
        url        = aceURL host port aceID aceBackend aceVariant
        act        = 10
        obs        = 32 -- 152
