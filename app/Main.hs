{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Lib hiding (info)
import qualified SAC
import qualified SAC.Defaults as SAC
import qualified TD3
import qualified TD3.Defaults as TD3
import qualified PPO
import qualified PPO.Defaults as PPO
import qualified MLFlow as MLF

import Control.Monad
import Options.Applicative

-- | Run Training
run :: Args -> IO ()
run Args{..} 
    | notElem algorithm $ map show [SAC.algorithm, TD3.algorithm, PPO.algorithm] 
            = error $ "No such algorithm " ++ algorithm
    | play = do
        when (algorithm == show SAC.algorithm) do
            let iter = SAC.numIterations
            SAC.loadAgent path obs act iter >>= SAC.play url uri
     
        when (algorithm == show TD3.algorithm) do
            let iter = TD3.numIterations
            TD3.loadAgent path obs act iter >>= TD3.play url uri
     
        when (algorithm == show PPO.algorithm) do
            let iter = PPO.numIterations
            PPO.loadAgent path obs act iter >>= PPO.play url uri

        putStrLn $ algorithm ++ " Agent finished playing episode."
    | otherwise = do
        putStrLn $ "Trainig " ++ algorithm ++ " Agent."

        when (algorithm == show SAC.algorithm) do
            SAC.train obs act url uri >>= SAC.saveAgent path'
     
        when (algorithm == show TD3.algorithm) do
            TD3.train obs act url uri >>= TD3.saveAgent path'
     
        when (algorithm == show PPO.algorithm) do
            PPO.train obs act url uri >>= PPO.saveAgent path'

        putStrLn $ "Training " ++ algorithm ++ " Agent finished."
      where
        path' = path ++ "/"  ++ algorithm
        url   = aceURL host port ace pdk var
        uri   = MLF.trackingURI mlfHost mlfPort

-- | Main
main :: IO ()
main = run =<< execParser opts
  where
    opts = info (args <**> helper) ( fullDesc <> progDesc "GACE RL Trainer" 
                                              <> header   "EDELWAC²E" )

-- | Command Line Arguments
data Args = Args { algorithm :: String
                 , host      :: String
                 , port      :: String
                 , ace       :: String
                 , pdk       :: String
                 , var       :: String
                 , act       :: Int
                 , obs       :: Int
                 , path      :: String
                 , mlfHost   :: String
                 , mlfPort   :: String
                 , play      :: Bool
                 } deriving (Show)

-- | Command Line Argument Parser
args :: Parser Args
args = Args <$> strOption ( long "algorithm" 
                         <> short 'l'
                         <> metavar "ALGORITHM" 
                         <> showDefault 
                         <> value "sac"
                         <> help "DRL Algorithm, one of sac, td3, ppo" )
            <*> strOption ( long "host" 
                         <> short 'H'
                         <> metavar "HOST" 
                         <> showDefault 
                         <> value "localhost"
                         <> help "Hym server host address" )
            <*> strOption ( long "port" 
                         <> short 'P'
                         <> metavar "PORT" 
                         <> showDefault 
                         <> value "7009"
                         <> help "Hym server port" )
            <*> strOption ( long "ace" 
                         <> short 'i'
                         <> metavar "ID" 
                         <> showDefault 
                         <> value "op2"
                         <> help "ACE OP ID" )
            <*> strOption ( long "pdk" 
                         <> short 'p'
                         <> metavar "PDK" 
                         <> showDefault 
                         <> value "xh035"
                         <> help "ACE Backend" )
            <*> strOption ( long "var" 
                         <> short 'v'
                         <> metavar "VARIANT" 
                         <> showDefault 
                         <> value "0"
                         <> help "GACE Environment Variant" )
            <*> option auto ( long "act" 
                           <> short 'a'
                           <> metavar "ACTIONS" 
                           <> showDefault 
                           <> value 10
                           <> help "Dimensions of Action Space" )
            <*> option auto ( long "obs" 
                           <> short 'o'
                           <> metavar "OBSERVATIONS" 
                           <> showDefault 
                           <> value 42
                           <> help "Dimensions of Observation Space" )
            <*> strOption ( long "path" 
                         <> short 'f'
                         <> metavar "FILE" 
                         <> showDefault 
                         <> value "./models"
                         <> help "Checkpoint File Path" )
            <*> strOption ( long "tracking-host" 
                         <> short 'T'
                         <> metavar "HOST" 
                         <> showDefault 
                         <> value "localhost"
                         <> help "MLFlow tracking server host address" )
            <*> strOption ( long "tracking-port" 
                         <> short 'R'
                         <> metavar "PORT" 
                         <> showDefault 
                         <> value "6008"
                         <> help "MLFlow tracking server port" )
            <*> switch ( long "play"
                      <> short 'y'
                      <> help "Play instead of training" )
