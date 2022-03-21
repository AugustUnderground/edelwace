{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Lib                  hiding (info)
import qualified SAC
import qualified TD3
import qualified PPO

import Control.Monad
import Options.Applicative

-- | Run Training
run :: Args -> IO ()
run Args{..} | algorithm `notElem` [SAC.algorithm, TD3.algorithm, PPO.algorithm] 
                         = error $ "No such algorithm " ++ algorithm
             | otherwise = do
    putStrLn $ "``'-.,_,.-'``'-.,_,.='``'-., Trainig " 
              ++ algorithm ++ " ,.-'``'-.,_,.='``'-.,_,.='``"

    when (algorithm == SAC.algorithm) do
        SAC.train obs act url >>= SAC.saveAgent path
 
    when (algorithm == TD3.algorithm) do
        TD3.train obs act url >>= TD3.saveAgent path
 
    when (algorithm == PPO.algorithm) do
        PPO.train obs act url >>= PPO.saveAgent path

    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"
  where
    url        = aceURL host port ace pdk var

-- | Main
main :: IO ()
main = run =<< execParser opts
  where
    opts = info (args <**> helper) ( fullDesc <> progDesc "GACE RL Trainer" 
                                              <> header "EDELWAC²E" )

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
                 } deriving (Show)

-- | Command Line Argument Parser
args :: Parser Args
args = Args <$> strOption ( long "algorithm" <> short 'l'
                                             <> metavar "ALGORITHM" 
                                             <> showDefault 
                                             <> value "sac"
                                             <> help "DRL Algorithm, one of sac, td3, ppo" )
            <*> strOption ( long "host" <> short 'H'
                                        <> metavar "HOST" 
                                        <> showDefault 
                                        <> value "localhost"
                                        <> help "Hym server host address" )
            <*> strOption ( long "port" <> short 'P'
                                        <> metavar "PORT" 
                                        <> showDefault 
                                        <> value "7009"
                                        <> help "Hym server port" )
            <*> strOption ( long "ace" <> short 'i'
                                       <> metavar "ID" 
                                       <> showDefault 
                                       <> value "op2"
                                       <> help "ACE OP ID" )
            <*> strOption ( long "pdk" <> short 'p'
                                       <> metavar "PDK" 
                                       <> showDefault 
                                       <> value "xh035"
                                       <> help "ACE Backend" )
            <*> strOption ( long "var" <> short 'v'
                                       <> metavar "VARIANT" 
                                       <> showDefault 
                                       <> value "0"
                                       <> help "GACE Environment Variant" )
            <*> option auto ( long "act" <> short 'a'
                                         <> metavar "ACTIONS" 
                                         <> showDefault 
                                         <> value 10
                                         <> help "Dimensions of Action Space" )
            <*> option auto ( long "obs" <> short 'o'
                                         <> metavar "OBSERVATIONS" 
                                         <> showDefault 
                                         <> value 40
                                         <> help "Dimensions of Observation Space" )
            <*> strOption ( long "path" <> short 'f'
                                        <> metavar "FILE" 
                                        <> showDefault 
                                        <> value "./models"
                                        <> help "Checkpoint File Path" )
