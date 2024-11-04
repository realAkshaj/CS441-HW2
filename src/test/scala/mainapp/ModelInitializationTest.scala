package mainapp

import modelgenerator.SlidingWindowModelTraining
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}

class ModelInitializationTest extends AnyFlatSpec with Matchers {

  "Model initialization" should "create a model with correct layer configuration" in {
    val inputSize = 194   // Replace with expected input size
    val hiddenSize = 64   // Hidden layer size used in configuration
    val outputSize = 50   // Replace with expected output size

    // Initialize the model
    val model = SlidingWindowModelTraining.initializeModel(inputSize, hiddenSize, outputSize)

    // Verify input and output sizes for each layer by casting to the appropriate type
    val layer0 = model.getLayer(0).getConfig.asInstanceOf[DenseLayer]
    layer0.getNIn shouldEqual inputSize
    layer0.getNOut shouldEqual hiddenSize

    val layer1 = model.getLayer(1).getConfig.asInstanceOf[OutputLayer]
    layer1.getNIn shouldEqual hiddenSize
    layer1.getNOut shouldEqual outputSize
  }
}
