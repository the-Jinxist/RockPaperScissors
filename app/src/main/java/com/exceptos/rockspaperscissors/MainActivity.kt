package com.exceptos.rockspaperscissors

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val outputs = TensorBuffer.createFixedSize(IntArray(16), DataType.FLOAT32)


    }
}
