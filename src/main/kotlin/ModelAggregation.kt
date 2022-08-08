import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import java.io.File

object ModelAggregation {
    val filePathList = ArrayList<String>()

    fun aggregation(layer: Int, alpha: Double) {
        if (filePathList.isEmpty()) return
        val originModel: MultiLayerNetwork
        try {
            val file = File("res/model/trained_model.zip")
            originModel = ModelSerializer.restoreMultiLayerNetwork(file, true)
            file.delete()
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

        for (filePath in filePathList) {
            try {
                val file = File(filePath)
                file.delete()
            } catch (e: Exception) {
                Utils.log(e.message.toString())
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