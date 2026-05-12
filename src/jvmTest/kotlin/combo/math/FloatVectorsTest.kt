package combo.math

class FloatVectorTest : VectorTest(FloatVectorFactory)
class SparseFloatVectorTest : VectorTest(SparseFloatVectorFactory)
class FloatMatrixTest : MatrixTest(FloatVectorFactory)

object SparseFloatVectorFactory : VectorFactory {
    override fun zeroVector(size: Int) = FloatSparseVector(size, FloatArray(size), IntArray(size) { it })
    override fun vector(values: FloatArray) = FloatSparseVector(values.size, values, IntArray(values.size) { it })
    override fun sparseVector(size: Int, values: FloatArray, indices: IntArray) = FloatSparseVector(size, values, indices)
    override fun matrix(values: Array<FloatArray>) = FloatVectorFactory.matrix(values)
    override fun zeroMatrix(rows: Int, columns: Int) = FloatVectorFactory.zeroMatrix(rows, columns)
}