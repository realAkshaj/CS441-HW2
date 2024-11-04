package mainapp

import modelgenerator.SlidingWindowModelTraining
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

class ModelTrainingTest extends AnyFlatSpec with Matchers {

  "Model training" should "reduce the loss on a small in-memory dataset" in {

    // Define a small sample dataset with inputs reshaped to 2D (1, inputSize)
    val sampleData = Seq(
      new DataSet(Nd4j.create(Array(1.0, 2.0, 3.0, 4.0, 5.0)).reshape(1, 5), Nd4j.create(Array(0.1)).reshape(1, 1)),
      new DataSet(Nd4j.create(Array(2.0, 3.0, 4.0, 5.0, 6.0)).reshape(1, 5), Nd4j.create(Array(0.2)).reshape(1, 1)),
      new DataSet(Nd4j.create(Array(3.0, 4.0, 5.0, 6.0, 7.0)).reshape(1, 5), Nd4j.create(Array(0.3)).reshape(1, 1)),
      new DataSet(Nd4j.create(Array(4.0, 5.0, 6.0, 7.0, 8.0)).reshape(1, 5), Nd4j.create(Array(0.4)).reshape(1, 1))
    )

    // Initialize model with learning rate in configuration
    val inputSize = 5
    val hiddenSize = 3
    val outputSize = 1
    val learningRate = 0.001

    val modelConfig = SlidingWindowModelTraining.initializeModelConfig(inputSize, hiddenSize, outputSize, learningRate)
    val model = new MultiLayerNetwork(modelConfig)
    model.init()
    model.setListeners(new ScoreIterationListener(1)) // Log score every iteration for feedback

    // Initial score (loss)
    val initialLoss = sampleData.map(ds => model.score(ds)).sum / sampleData.size

    // Train for a few epochs in-memory
    for (_ <- 1 to 5) {
      sampleData.foreach(model.fit) // Fit each data point
    }

    // Final score (loss) after training
    val finalLoss = sampleData.map(ds => model.score(ds)).sum / sampleData.size

    println(s"Initial Loss: $initialLoss, Final Loss: $finalLoss")

    // Verify that the loss has decreased
    assert(finalLoss < initialLoss, "Final loss should be lower than initial loss after training")
  }
}
