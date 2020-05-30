package com.exceptos.rockspaperscissors

import android.util.Log
import kotlin.math.sqrt

class Calculations {

    companion object{

        private val TAG = "Calculations"

        fun findSmallestDistance(floatList: List<Float>): Float{
            //Finding the smallest value in the array
            var smallestFloat = floatList[0]
            for (float in floatList){
                if (float < smallestFloat) smallestFloat = float
            }
            return smallestFloat
        }

        fun calculateEuclideanDistance(detectionOutputArray: FloatArray, vectorArray: FloatArray): Float{
            //The formula: sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
            var sum = 0f
            if (detectionOutputArray.size == vectorArray.size){
                val size = detectionOutputArray.size -1
                for(i in 0..size){
                    sum += (detectionOutputArray[i] - vectorArray[i]) * (detectionOutputArray[i] - vectorArray[i])
                }
            }else{
                Log.e(TAG, "Euclidean Calculation: The lengths of output vectors and test vectors differ")

            }

            return sqrt(sum)
        }
    }



}