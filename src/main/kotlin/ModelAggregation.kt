import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
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

    fun aggregation(layer: Int, alpha: Double) {
        if (filePathList.isEmpty()) return
        val originModel: MultiLayerNetwork
        try {
            val file = File("res/model/trained_model.zip")
            originModel = ModelSerializer.restoreMultiLayerNetwork(file, true)
        } catch (e: Exception) {
            Utils.log(e.message.toString())
            return
        }
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