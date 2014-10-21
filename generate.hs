module Main where

import Control.Arrow (second)

values = takeWhile (<20) [0,0.75..]

circle_center = (10,10)
circle_rad   = 7
plane = concat $ map (\v -> zip (repeat v) values) values

inside :: (Float,Float) -> Float -> (Float,Float) -> Bool
inside (cx,cy) rad (x,y) = (x - cx) ^ 2 + (y - cy) ^ 2 <= rad ^ 2

check :: [((Float,Float),Bool)]
check = map func plane
    where
        func c = (c, (inside circle_center circle_rad c))

fromBool :: Bool -> Float
fromBool True  = 1
fromBool False = 0

fromBool' :: Bool -> Float
fromBool' True  = 1
fromBool' False = -1

main :: IO ()
main = mapM_ putStrLn $ generate fromBool'

generate :: (Bool -> Float) -> [String]
generate f = map (showTuple . second f) check
    where
        showTuple ((x,y),t) = show x ++ " " ++ show y ++ " " ++ show t

