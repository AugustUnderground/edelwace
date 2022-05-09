import qualified Torch as T
import qualified Torch.Vision as T (randomIndexes)
import qualified RPB.HER as HER
import qualified Data.Set as S
import qualified Data.Map as M

main :: IO ()
main = putStrLn "Test suite not yet implemented"

am <- tensorToMap <$> T.randnIO' [2,10]
am = tensorToMap $ T.zeros' [2,10]

sm <- hymPoolStep envUrl am
(s',r,d,_) = stepsToTuple'  sm

