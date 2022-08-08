import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import kotlin.concurrent.thread

object ModelAggregation {
    val filePathList = ArrayList<String>()
    private lateinit var model: MultiLayerNetwork
    private val uiServer = UIServer.getInstance()
    private val statsStorage = InMemoryStatsStorage()
    private var trainingData: DataSet

    init {
        try {
            val file = File("res/model/trained_model.zip")
            model = ModelSerializer.restoreMultiLayerNetwork(file, true)
        } catch (e: Exception) {
            Utils.log(e.message.toString())
        }

        val row = 150
        val col = 4
        val irisMatrix = Array(row) { DoubleArray(col) }
        var i = 0
        for (r in 0 until row) {
            for (c in 0 until col) {
                irisMatrix[r][c] = TrainingData.irisData[i++]
            }
        }
        val rowLabel = 150
        val colLabel = 3
        val twodimLabel = Array(rowLabel) { DoubleArray(colLabel) }
        i = 0
        for (r in 0 until rowLabel) {
            for (c in 0 until colLabel) {
                twodimLabel[r][c] = TrainingData.labelData[i++]
            }
        }
        val trainingIn = Nd4j.create(irisMatrix)
        val trainingOut = Nd4j.create(twodimLabel)
        trainingData = DataSet(trainingIn, trainingOut)
    }

    fun startWebUI() {
        model.setListeners(StatsListener(statsStorage, 1))
        uiServer.attach(statsStorage)
        Utils.log("Web UI started on http://localhost:9000/")
    }

    fun createModel() {
        val inputLayer = DenseLayer.Builder()
            .nIn(4)
            .nOut(20)
            .name("Input")
            .build()

        //隐藏层
        val hiddenLayer = DenseLayer.Builder()
            .nIn(20)
            .nOut(20)
            .name("Hidden")
            .build()

        //输出层
        val outputLayer = OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nIn(20)
            .nOut(3)
            .name("Output")
            .activation(Activation.SOFTMAX) //标准的 softmax 激活函数
            .build()

        val nncBuilder = NeuralNetConfiguration.Builder()
        val seed = 6L
        nncBuilder.seed(seed)
        nncBuilder.activation(Activation.TANH) //标准的双曲正切激活函数
        nncBuilder.weightInit(WeightInit.XAVIER) //初始权重为均值 0, 方差为 2.0/(fanIn + fanOut)的高斯分布

        val listBuilder = nncBuilder.list()
        listBuilder.layer(0, inputLayer)
        listBuilder.layer(1, hiddenLayer)
        listBuilder.layer(2, outputLayer)

        model = MultiLayerNetwork(listBuilder.build())
        model.init()

        Utils.log("Model created")
    }

    fun aggregation(layer: Int, alpha: Double) {
        if (filePathList.isEmpty()) return
        val originModel = model
        val filePathList = ArrayList(this.filePathList)
        this.filePathList.clear()

        for (i in 0 until layer) {
            var paramTable = originModel.paramTable()
            var weights = paramTable[String.format("%d_W", i)]!!
            var bias = paramTable[String.format("%d_b", i)]!!
            var avgWeights = weights.mul(alpha)
            var avgBias = bias.mul(alpha)

            for (filePath in filePathList) {
                var model: MultiLayerNetwork
                try {
                    val file = File(filePath)
                    model = ModelSerializer.restoreMultiLayerNetwork(file, true)
                } catch (e: Exception) {
                    Utils.log(e.message.toString())
                    continue
                }
                paramTable = model.paramTable()
                weights = paramTable[String.format("%d_W", i)]!!
                bias = paramTable[String.format("%d_b", i)]!!
                avgWeights = avgWeights.add(weights.mul(1.0 - alpha).div(filePathList.size))
                avgBias = avgBias.add(bias.mul(1.0 - alpha).div(filePathList.size))
            }

            originModel.setParam(String.format("%d_W", i), avgWeights)
            originModel.setParam(String.format("%d_b", i), avgBias)
        }

        Utils.log("Successfully aggregated ${filePathList.size} models")

        thread {
            try {
                val paramTable = originModel.paramTable()
                for (i in 0 until layer) {
                    model.setParam(String.format("%d_W", i), paramTable[String.format("%d_W", i)])
                    model.setParam(String.format("%d_b", i), paramTable[String.format("%d_b", i)])
                }
                model.fit(trainingData)
            } catch (e: Exception) {
                Utils.log(e.message.toString())
            }
        }

        thread {
            for (filePath in filePathList) {
                try {
                    val file = File(filePath)
                    file.delete()
                } catch (e: Exception) {
                    Utils.log(e.message.toString())
                }
            }
        }

        try {
            ModelSerializer.writeModel(originModel, "res/model/trained_model.zip", true)
            Utils.log("Aggregated model saved")
        } catch (e: Exception) {
            Utils.log(e.message.toString())
        }
    }
}