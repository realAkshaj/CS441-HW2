package modelgenerator

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

class ModelFileSaveTest extends AnyFlatSpec with Matchers {

  "saveModel" should "create a file at the specified path" in {
    // Initialize model parameters for testing
    val inputSize = 10
    val hiddenSize = 20
    val outputSize = 5
    val model: MultiLayerNetwork = SlidingWindowModelTraining.initializeModel(inputSize, hiddenSize, outputSize)

    // Define a temporary file path for saving the model
    val filePath = "test_model.zip"

    // Save the model
    SlidingWindowModelTraining.saveModel(model, filePath)

    // Check that the file exists
    val file = new File(filePath)
    file.exists() shouldEqual true

    // Clean up: delete the file after test
    file.delete()
  }
}

