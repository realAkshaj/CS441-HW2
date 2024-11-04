package modelgenerator

import modelgenerator.SlidingWindowModelTraining
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SlidingWindowModelTrainingTest extends AnyFlatSpec with Matchers {

  "initializeModel" should "create a model with the correct layer sizes" in {
    val inputSize = 10
    val hiddenSize = 20
    val outputSize = 5

    val model: MultiLayerNetwork = SlidingWindowModelTraining.initializeModel(inputSize, hiddenSize, outputSize)

    val inputLayer = model.getLayer(0).getConfig.asInstanceOf[DenseLayer]
    inputLayer.getNIn shouldEqual inputSize
    inputLayer.getNOut shouldEqual hiddenSize

    val outputLayer = model.getLayer(1).getConfig.asInstanceOf[OutputLayer]
    outputLayer.getNIn shouldEqual hiddenSize
    outputLayer.getNOut shouldEqual outputSize
  }
}
