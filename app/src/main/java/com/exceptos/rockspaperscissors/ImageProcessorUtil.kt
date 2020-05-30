package com.exceptos.rockspaperscissors

import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp

class ImageProcessorUtil {

    companion object{

        private val TAG = "ImageProcessor"

        private const val RESIZE_WIDTH = 300
        private const val RESIZE_HEIGHT = 300

        private const val NORMALIZE_MEAN = 255f
        private const val NORMALIZE_STD = 255F

        fun processImage(bitmap: Bitmap
        ): TensorImage {

            //To properly get a buffer from the bitmap
            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            val imageProcessor = ImageProcessor.Builder()
                    //Resizing to the specified 300 * 300
                    .add(ResizeOp(RESIZE_HEIGHT, RESIZE_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                    .add(ResizeWithCropOrPadOp(RESIZE_HEIGHT, RESIZE_WIDTH))
                    .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD))
                    //Normalizing my dividing it with 255f
                    .add(CastOp(DataType.FLOAT32))
                    //Casting the resulting data to float
                    .build()

            Log.e(TAG, "Processing Image..")

            return imageProcessor.process(tensorImage)
        }

    }

}