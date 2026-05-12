@file:JvmName("Arrays")

package combo.util

import kotlin.jvm.JvmName

inline fun IntArray.sumByFloat(selector: (Int) -> Float): Float {
    var sum = 0.0f
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

inline fun FloatArray.sumByFloat(selector: (Float) -> Float): Float {
    var sum = 0.0f
    for (element in this) {
        sum += selector(element)
    }
    return sum
}

inline fun FloatArray.mapArray(transform: (Float) -> Float) =
    FloatArray(this.size) {
        transform(this[it])
    }

inline fun IntArray.mapArray(transform: (Int) -> Int) =
    IntArray(this.size) {
        transform(this[it])
    }

inline fun <T1, reified T2> Array<T1>.mapArray(transform: (T1) -> T2) =
    Array(this.size) {
        transform(this[it])
    }

inline fun FloatArray.mapArrayIndexed(transform: (Int, Float) -> Float) =
    FloatArray(this.size) {
        transform(it, this[it])
    }

inline fun FloatArray.transformArrayIndexed(transform: (Int, Float) -> Float) {
    for (i in indices)
        this[i] = transform(i, this[i])
}

inline fun IntArray.transformArrayIndexed(transform: (Int, Int) -> Int) {
    for (i in indices)
        this[i] = transform(i, this[i])
}

inline fun FloatArray.transformArray(transform: (Float) -> Float) {
    for (i in indices)
        this[i] = transform(this[i])
}

inline fun IntArray.transformArray(transform: (Int) -> Int) {
    for (i in indices)
        this[i] = transform(this[i])
}

fun FloatArray.removeAt(ix: Int) = copyOfRange(0, ix) + copyOfRange(ix + 1, size)
fun IntArray.removeAt(ix: Int) = copyOfRange(0, ix) + copyOfRange(ix + 1, size)
fun LongArray.removeAt(ix: Int) = copyOfRange(0, ix) + copyOfRange(ix + 1, size)
fun <T> Array<T>.removeAt(ix: Int) = copyOfRange(0, ix) + copyOfRange(ix + 1, size)
val EMPTY_INT_ARRAY = IntArray(0)
val EMPTY_FLOAT_ARRAY = FloatArray(0)
val EMPTY_LONG_ARRAY = LongArray(0)
