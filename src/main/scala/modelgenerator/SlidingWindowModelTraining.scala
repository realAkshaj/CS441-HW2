package modelgenerator

import org.apache.spark.SparkContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

object SlidingWindowModelTraining {

  def run(args: Array[String]): Unit = {

    // Set up Spark context
    val conf = new SparkConf().setAppName("SlidingWindowModelTraining").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // Configurable parameters
    val batchSize = 32
    val numEpochs = 5
    val modelOutputPath = "src/main/finalOutput/decoder_model.zip"
    val dataPath = "src/main/correctedInput/sliding_window_dataset/sliding_window_dataset.csv"

    // Helper function to parse a space-separated string of doubles
    def parseDoubles(str: String): Array[Double] = {
      str.split(" ").flatMap { s =>
        try {
          Some(s.toDouble)
        } catch {
          case _: NumberFormatException => None
        }
      }
    }

    // Load and parse dataset from CSV
    val dataRDD: RDD[DataSet] = sc.textFile(dataPath).flatMap { line =>
      val parts = line.split(",")
      if (parts.length == 2) {
        val inputWindow = parseDoubles(parts(0))
        val target = parseDoubles(parts(1))

        // Ensure data is in correct format
        if (inputWindow.nonEmpty && target.nonEmpty) {
          val input = Nd4j.create(inputWindow).reshape(1, inputWindow.length)
          val label = Nd4j.create(target).reshape(1, target.length)
          Some(new DataSet(input, label))
        } else None
      } else None
    }

    // Prepare JavaRDD for training
    val trainingJavaRDD = dataRDD.toJavaRDD()

    // Model parameters
    val inputSize = trainingJavaRDD.first().getFeatures.size(1).toInt
    val outputSize = trainingJavaRDD.first().getLabels.size(1).toInt
    val hiddenSize = 64

    // Build the model
    val model = new MultiLayerNetwork(
      new NeuralNetConfiguration.Builder()
        .list()
        .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(hiddenSize).activation(Activation.RELU).build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
          .nIn(hiddenSize)
          .nOut(outputSize)
          .activation(Activation.IDENTITY)
          .build())
        .build()
    )
    model.init()
    model.setListeners(new ScoreIterationListener(10))

    // Training master for distributed training
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .build()

    // Spark DL4J Model for distributed training
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Train the model
    for (epoch <- 1 to numEpochs) {
      sparkModel.fit(trainingJavaRDD)
      println(s"Completed epoch $epoch")
    }

    // Save the trained model
    ModelSerializer.writeModel(sparkModel.getNetwork, new File(modelOutputPath), true)
    println(s"Model training complete and saved to $modelOutputPath")

    // Stop Spark Context
    sc.stop()
  }
}













