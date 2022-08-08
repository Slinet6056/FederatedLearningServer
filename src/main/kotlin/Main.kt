fun main(args: Array<String>) {
    val socketServer = SocketServer()
    socketServer.startServer(12345)
    ModelAggregation.createModel()
    ModelAggregation.startWebUI()
    while (true) {
        when (readln()) {
            "stop" -> {
                socketServer.stopServer()
                break
            }

            "send" -> {
                socketServer.sendFile("res/model/trained_model.zip")
            }

            "avg" -> {
                ModelAggregation.aggregation(3, 0.3)
            }
        }
    }
}