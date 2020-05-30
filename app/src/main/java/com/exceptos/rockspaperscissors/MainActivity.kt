package com.exceptos.rockspaperscissors

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.exceptos.rockspaperscissors.Calculations.Companion.calculateEuclideanDistance
import com.exceptos.rockspaperscissors.Calculations.Companion.findSmallestDistance
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import java.io.BufferedReader
import java.io.FileNotFoundException
import java.io.IOException
import java.io.InputStreamReader
import java.nio.MappedByteBuffer

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

                val processedImage = ImageProcessorUtil.processImage(imageBitmap!!)
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

        //Initializing variables
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


    private fun detectHandSign(image: TensorImage): FloatArray{

//        float[][] result = new float[1][labelList.size()];

        val interpreter = Interpreter(modelBuffer)

        //The output is said to be an array containing 16 floats
        val result = arrayOf(FloatArray(16))
        interpreter.run(image.buffer, result)

        //Return the output float array
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


    /**
     *
     * @param vectorOutputString Contains lines of vectors separated by tabs which will be used to test the output
     * @param labelOutputString Contains each label by index
     * @param detectionOutputArray The output we which to identify
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
                Log.e(this.javaClass.simpleName, "Vector Array: $${vectorArray[14]}")
                val distance = calculateEuclideanDistance(detectionOutputArray, vectorArray)

                //Storing this euclidean distance in a list
                Log.e(this.javaClass.simpleName, "Distance: $distance")
                listOfEuclideanDistances.add(distance)
            }

            //Finding the minimum euclidean distance
            val smallestFloat = findSmallestDistance(listOfEuclideanDistances)
            Log.e(this.javaClass.simpleName, "Smallest distance: $smallestFloat")

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

        val alertDialog = AlertDialog.Builder(this)
        alertDialog.setCancelable(true)

        when(label){
            "0" -> {
                alertDialog.setMessage("This hand sign is Rock!!")
                alertDialog.show()

            }
            "1" -> {
                alertDialog.setMessage("This hand sign is Paper!!")
                alertDialog.show()

            }
            "2" -> {
                alertDialog.setMessage("This hand sign is Scissors!!")
                alertDialog.show()
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
