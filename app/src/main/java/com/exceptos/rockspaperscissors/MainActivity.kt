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
        }else if(p0 == takePictureButton){
            if (imageBitmap != null){
                val processedImage = processImage(imageBitmap!!)
                detectHandSign((processedImage))
            }
        }
    }

    private lateinit var outputs: TensorBuffer
    private lateinit var modelBuffer: MappedByteBuffer
    private var imageBitmap: Bitmap? = null

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

        takePictureButton.setOnClickListener(this)
        detectHandSignButton.setOnClickListener(this)

        loadCSVFiles()

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

        return imageProcessor.process(tensorImage)
    }

    private fun detectHandSign(image: TensorImage): IntArray{
        val interpreter = Interpreter(modelBuffer)
        interpreter.run(image.tensorBuffer.buffer, outputs.buffer)
        Log.e(this.javaClass.simpleName, "Output: ${outputs.intArray}")
        Toast.makeText(this, "Output: ${outputs.intArray}", Toast.LENGTH_LONG)
            .show()
        return outputs.intArray
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

    fun performEuclideanDistanceCalculation(vectorOutputString: MutableList<String>, labelOutputString: MutableList<String>){

        val listOfEuclideanDistances = mutableListOf<Float>()

        if (vectorOutputString.size == labelOutputString.size){
            for(vectorString in vectorOutputString){
                vectorString.split('\t').map { it.toFloat() }
            }
        }else{
            Log.e(this.javaClass.simpleName, "Euclidean Calculation: The lengths differ")
        }
    }

    private fun loadCSVFiles(){

        val vectorsTsvFile = BufferedReader(InputStreamReader(assets.open("vecs.tsv")))
        var vectorDataRow = vectorsTsvFile.readLine()

        val vectorOutputString = mutableListOf<String>()
        while (vectorDataRow != null){
            vectorOutputString.add(vectorDataRow)
            Log.e(this.javaClass.simpleName, "Test With: $vectorOutputString")
            vectorDataRow = vectorsTsvFile.readLine()
        }
        vectorsTsvFile.close()

        val labelsTsvFile = BufferedReader(InputStreamReader(assets.open("labels.tsv")))
        var labelsDataRow = labelsTsvFile.readLine()

        val labelsOutputString = mutableListOf<String>()
        while (labelsDataRow != null){
            labelsOutputString.add(labelsDataRow)
            Log.e(this.javaClass.simpleName, "Test With: $labelsOutputString")
            labelsDataRow = labelsTsvFile.readLine()
        }
        labelsTsvFile.close()

    }

}
