import java.text.SimpleDateFormat
import java.util.*

object Utils {
    private fun currentTime(): String {
        val sdf = SimpleDateFormat()
        sdf.applyPattern("yyyy-MM-dd HH:mm:ss a")
        val date = Date()
        return sdf.format(date)
    }

    fun log(msg: String) {
        println("${currentTime()}: $msg")
    }
}