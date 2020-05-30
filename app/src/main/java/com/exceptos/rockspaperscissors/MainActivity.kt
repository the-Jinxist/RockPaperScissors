package com.exceptos.rockspaperscissors

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.nio.MappedByteBuffer
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), View.OnClickListener {

    override fun onClick(p0: View?) {
        if (p0!! == takePictureButton){
            val cameraIntent =  Intent(
                Intent.ACTION_PICK
            )
            cameraIntent.type = "image/*"
            if (cameraIntent.resolveActivity(packageManager) != null){
                startActivityForResult(cameraIntent, PICK_USER_PROFILE_IMAGE)
            }
        }else if(p0 == detectHandSignButton){
            if (imageBitmap != null){
                Log.e(this.javaClass.simpleName, "Detection started")

                val processedImage = processImage(imageBitmap!!)
                val outputArray = detectHandSign((processedImage))

                if (mVectorOutputString.isNotEmpty() && mLabelOutputString.isNotEmpty()){
                    performEuclideanDistanceCalculation(
                            vectorOutputString = mVectorOutputString,
                            labelOutputString = mLabelOutputString,
                            detectionOutputArray = outputArray
                    )
                }

            }
        }
    }

    private lateinit var outputs: TensorBuffer
    private lateinit var modelBuffer: MappedByteBuffer
    private var imageBitmap: Bitmap? = null
    private lateinit var mVectorOutputString: MutableList<String>
    private lateinit var mLabelOutputString: MutableList<String>

    private val PICK_USER_PROFILE_IMAGE = 10

    private lateinit var detectHandSignButton: Button
    private lateinit var takePictureButton: Button
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        outputs = TensorBuffer.createFixedSize(IntArray(16), DataType.FLOAT32)
        modelBuffer = FileUtil.loadMappedFile(this, "model.tflite")

        detectHandSignButton = findViewById(R.id.detect_hand_sign)
        detectHandSignButton.isEnabled = false
        takePictureButton = findViewById(R.id.take_picture)
        imageView = findViewById(R.id.image_view)

        mVectorOutputString = mutableListOf()
        mLabelOutputString = mutableListOf()

        takePictureButton.setOnClickListener(this)
        detectHandSignButton.setOnClickListener(this)

        loadTSVFiles()

    }

    private fun processImage(bitmap: Bitmap
    ): TensorImage{

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
            .add(ResizeWithCropOrPadOp(300, 300))
            .add(NormalizeOp(255f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()

        Log.e(this.javaClass.simpleName, "Processing Image..")

        return imageProcessor.process(tensorImage)
    }

    private fun detectHandSign(image: TensorImage): FloatArray{

//        float[][] result = new float[1][labelList.size()];

        val interpreter = Interpreter(modelBuffer)
        val result = arrayOf(FloatArray(16))
        interpreter.run(image.buffer, result)
        Log.e(this.javaClass.simpleName, "Output: ${result[0][6]}")
        Toast.makeText(this, "Output: ${result[0][3]}", Toast.LENGTH_LONG)
            .show()
        return result[0]
    }

    private fun convertUriToBitmap(imageUri: Uri): Bitmap{
        var bitmap: Bitmap? = null
        try {
            val inputStream = contentResolver.openInputStream(imageUri)
            bitmap = BitmapFactory.decodeStream(inputStream)

            try {
                inputStream?.close()
            }catch (e: IOException){
                Log.e(this.javaClass.simpleName, "Bitmap Error: ${e.message!!}")
            }

        }catch (e: FileNotFoundException){
            Log.e(this.javaClass.simpleName, "Bitmap Error: ${e.message!!}")
        }
        return bitmap!!
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK){
            if (requestCode == PICK_USER_PROFILE_IMAGE){
                if (data != null && data.data !=  null){
                    detectHandSignButton.isEnabled = true

                    imageView.setImageURI(data.data!!)
                    imageBitmap = convertUriToBitmap(data.data!!)
                }

            }
        }
    }

    private fun findSmallestDistance(floatList: List<Float>): Float{
        //Finding the smallest value in the array
        var smallestFloat = floatList[0]
        for (float in floatList){
            if (float < smallestFloat) smallestFloat = float
        }
        return smallestFloat
    }

    private fun calculateEuclideanDistance(detectionOutputArray: FloatArray, vectorArray: FloatArray): Float{
        //The formula: sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
        var sum = 0f
        if (detectionOutputArray.size == vectorArray.size){
            for(i in (0..detectionOutputArray.size - 1)){
                sum += (detectionOutputArray[i] - vectorArray[i]) * (detectionOutputArray[i] - vectorArray[i])
            }
        }else{
            Log.e(this.javaClass.simpleName, "Euclidean Calculation: The lengths of output vectors and test vectors differ")

        }

        return sqrt(sum)
    }


    /**
     *
     * @param vectorOutputString Contains lines of vectors seperated by tabs which will be used to test the output
     * @param labelOutputString Contains each label by index
     * @param detectionOutputArray The ouput we which to identify
     */
    fun performEuclideanDistanceCalculation(
        vectorOutputString: MutableList<String>,
        labelOutputString: MutableList<String>,
        detectionOutputArray: FloatArray
        ){

        //First we initialize an empty list of floats which will contain all the
        //euclidean distances. The index of each euclidean distance will correspond to their respective
        //index in the list of labels we retrieved earlier
        val listOfEuclideanDistances = mutableListOf<Float>()

        if (vectorOutputString.size == labelOutputString.size){
            for(vectorString in vectorOutputString){
                //Taking each string, separating it be tabs and calculating the euclidean distance
                val vectorArray = vectorString.split('\t').map { it.toFloat() }.toFloatArray()
                Log.e(this.javaClass.simpleName, "Vector Array: $vectorArray")
                val distance = calculateEuclideanDistance(detectionOutputArray, vectorArray)
                //Storing this euclidean distance in a list
                listOfEuclideanDistances.add(distance)
            }

            //Finding the minimum euclidean distance
            val smallestFloat = findSmallestDistance(listOfEuclideanDistances)

            /**
            // Corresponding it to the index of the list of labels [labelOutputString]
            */
            val index = listOfEuclideanDistances.indexOf(smallestFloat)
            val label = labelOutputString[index]

            notifyHandSign(label)


        }else{
            Log.e(this.javaClass.simpleName, "Euclidean Calculation: The lengths of labels and test vectors differ")
        }
    }

    private fun notifyHandSign(label: String) {
        when(label){
            "0" -> {
                Toast.makeText(this, "This hand sign is Rock", Toast.LENGTH_LONG)
                    .show()
            }
            "1" -> {
                Toast.makeText(this, "This hand sign is Paper", Toast.LENGTH_LONG)
                    .show()
            }
            "2" -> {
                Toast.makeText(this, "This hand sign is Scissors", Toast.LENGTH_LONG)
                    .show()
            }
        }
    }

    private fun loadTSVFiles(){

        //Loading the vector TSV files from the assets. The vectors by which we test our
        //output

        val vectorsTsvFile = BufferedReader(InputStreamReader(assets.open("vecs.tsv")))
        var vectorDataRow = vectorsTsvFile.readLine()

        while (vectorDataRow != null){
            mVectorOutputString.add(vectorDataRow)
            vectorDataRow = vectorsTsvFile.readLine()

        }
        vectorsTsvFile.close()

        //Loading the vector TSV files from the assets. The vectors by which we determine the
        // category of our output

        val labelsTsvFile = BufferedReader(InputStreamReader(assets.open("labels.tsv")))
        var labelsDataRow = labelsTsvFile.readLine()


        while (labelsDataRow != null){
            mLabelOutputString.add(labelsDataRow)
            labelsDataRow = labelsTsvFile.readLine()
        }
        labelsTsvFile.close()

    }

}
