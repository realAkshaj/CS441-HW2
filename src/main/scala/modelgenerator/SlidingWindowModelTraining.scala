package modelgenerator

import org.apache.spark.SparkContext
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
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
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.ndarray.INDArray

import java.io.{BufferedWriter, File, FileWriter, OutputStream, OutputStreamWriter, PrintWriter}
import org.apache.hadoop.conf.Configuration
import org.nd4j.linalg.learning.config.Sgd

import java.util.{List, Map}

object SlidingWindowModelTraining {

  def run(args: Array[String]): Unit = {

    // Configure Spark and DL4J settings
    val environment = if (args.isEmpty) "local" else args(0).toLowerCase
    val (masterUrl, appName) = environment match {
      case "spark" => ("spark://localhost:7077", "CS441-HW2-Spark")
      case "aws" => ("yarn", "CS441-HW2-AWS")
      case _ => ("local[*]", "CS441-HW2-Local")
    }

    // Set up Spark context
    val conf = new SparkConf().setAppName("SlidingWindowModelTraining").setMaster(masterUrl)
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.config(conf).getOrCreate()

    // Configurable parameters
    val batchSize = 32
    val numEpochs = 5
    val localModelPath = "tmp/decoder_model.zip" // Local path to save the model temporarily
//    val hdfsModelPath = "hdfs:///user/akshaj/output/sliding_window_dataset/decoder_model.zip"       //local
//    val dataPath = "hdfs:///user/akshaj/output/sliding_window_dataset/sliding_window_dataset.csv"   //local
    val hdfsModelPath = "src/main/finalOutput/decoder_model.zip"      //aws
    val dataPath = "src/main/correctedInput/sliding_window_dataset/sliding_window_dataset.csv"   //aws
//    val hdfsModelPath = "hdfs://localhost:9000/output/user/aksha/sliding_window_dataset/decoder_model.zip" //spark
//    val dataPath = "hdfs://localhost:9000/output/user/aksha/sliding_window_dataset/sliding_window_dataset.csv" //spark

//    val metricsFilePath =  "hdfs:///user/akshaj/output/metrics.csv"
    val metricsFilePath =  "src/main/finalOutput/metrics.csv"

    // Prepare file for metrics output
    val metricsWriter = new BufferedWriter(new FileWriter(metricsFilePath))
    metricsWriter.write("Epoch,Iteration,Training Loss,Gradient Norm,Memory Usage (MB),Learning Rate,Duration (ms)\n")

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

    model.setListeners(new ScoreIterationListener(10), new GradientStatsListener(metricsWriter))

    // Training master for distributed training
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .build()
    // Spark DL4J Model for distributed training
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Training loop
    for (epoch <- 1 to numEpochs) {
      val epochStart = System.currentTimeMillis()

      sparkModel.fit(trainingJavaRDD)
      val epochEnd = System.currentTimeMillis()
      val epochDuration = epochEnd - epochStart

      val trainingLoss = model.score()
      val memoryUsed = (Runtime.getRuntime.totalMemory - Runtime.getRuntime.freeMemory) / (1024 * 1024)
      val learningRate = 0.001 // Set to your learning rate or retrieve dynamically if available

      metricsWriter.write(s"$epoch,-1,$trainingLoss,-1,$memoryUsed,$learningRate,$epochDuration\n")
      metricsWriter.flush()
    }

    // Save the trained model locally
//    ModelSerializer.writeModel(sparkModel.getNetwork, new File(localModelPath), true)
    val hdfsPath = new Path(hdfsModelPath)
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val outputStream: OutputStream = fs.create(hdfsPath, true)
    ModelSerializer.writeModel(sparkModel.getNetwork, outputStream, true)
    println(s"Model training complete and saved to $localModelPath")
    outputStream.close()
    fs.close()
    println(s"Model saved locally to $localModelPath")

    // Show Spark metrics
    println(s"Total executors: ${sc.getExecutorMemoryStatus.size}")
    sc.getExecutorMemoryStatus.foreach {
      case (executorId, (memUsed, memTotal)) =>
        println(s"Executor $executorId - Used memory: ${memUsed / (1024 * 1024)} MB, Total memory: ${memTotal / (1024 * 1024)} MB")
    }

    // Copy metrics file to HDFS or S3 if needed based on environment
//    val fs = FileSystem.get(new java.net.URI(hdfsModelPath), sc.hadoopConfiguration)
//    fs.copyFromLocalFile(new Path(metricsFilePath), new Path(hdfsModelPath.replace("decoder_model.zip", "metrics_log.csv")))
//    println(s"Metrics log copied to HDFS at ${hdfsModelPath.replace("decoder_model.zip", "metrics_log.csv")}")

    // Stop Spark Context and close writer
    metricsWriter.close()
    sc.stop()
  }

  // Custom listener for gradient statistics
  class GradientStatsListener(metricsWriter: BufferedWriter) extends TrainingListener {
    private var totalGradientNorm: Double = 0.0
    private var iterationCount: Int = 0

    override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
      if (model != null && model.gradient() != null) {
        val norm = model.gradient().gradient().norm2Number().doubleValue()
        totalGradientNorm += norm
        iterationCount += 1

        metricsWriter.write(s"$epoch,$iteration,-1,$norm,-1,-1,-1\n")
        metricsWriter.flush()
      }
    }

    override def onEpochEnd(model: Model): Unit = {
      if (iterationCount > 0) {
        val averageGradientNorm = totalGradientNorm / iterationCount
        metricsWriter.write(s"-1,-1,-1,$averageGradientNorm,-1,-1,-1\n")
        metricsWriter.flush()
      }
      totalGradientNorm = 0.0
      iterationCount = 0
    }

    // Empty overrides for other methods
    override def onGradientCalculation(model: Model): Unit = {}

    override def onForwardPass(model: Model, activations: java.util.Map[String, INDArray]): Unit = {}

    override def onForwardPass(model: Model, activations: java.util.List[INDArray]): Unit = {}

    override def onBackwardPass(model: Model): Unit = {}

    override def onEpochStart(model: Model): Unit = {}
  }
  def initializeModel(inputSize: Int, hiddenSize: Int, outputSize: Int): MultiLayerNetwork = {
    // Build and return the model
    val model = new MultiLayerNetwork(
      new NeuralNetConfiguration.Builder()
        .list()
        .layer(0, new DenseLayer.Builder()
          .nIn(inputSize)
          .nOut(hiddenSize)
          .activation(Activation.RELU)
          .build())
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
          .nIn(hiddenSize)
          .nOut(outputSize)
          .activation(Activation.IDENTITY)
          .build())
        .build()
    )
    model.init()
    model
  }
  def saveModel(model: MultiLayerNetwork, filePath: String): Unit = {
    ModelSerializer.writeModel(model, new File(filePath), true)
  }
  def initializeModelConfig(inputSize: Int, hiddenSize: Int, outputSize: Int, learningRate: Double): MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .updater(new Sgd(learningRate))
      .list()
      .layer(0, new DenseLayer.Builder().nIn(inputSize).nOut(hiddenSize).activation(Activation.RELU).build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(hiddenSize)
        .nOut(outputSize)
        .activation(Activation.IDENTITY)
        .build())
      .build()
  }

}

