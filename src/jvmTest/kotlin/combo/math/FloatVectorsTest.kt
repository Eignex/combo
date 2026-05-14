package combo.math

class FloatVectorTest : VectorTest(FloatVectorFactory)
class SparseFloatVectorTest : VectorTest(SparseFloatVectorFactory)
class FloatMatrixTest : MatrixTest(FloatVectorFactory)

object SparseFloatVectorFactory : VectorFactory {
    override fun zeroVector(size: Int) = FloatSparseVector(size, IntArray(size) { it }, FloatArray(size))
    override fun vector(values: FloatArray) = FloatSparseVector(values.size, IntArray(values.size) { it }, values)
    override fun sparseVector(size: Int, values: FloatArray, indices: IntArray) = FloatSparseVector(size, indices, values)
    override fun matrix(values: Array<FloatArray>) = FloatVectorFactory.matrix(values)
    override fun zeroMatrix(rows: Int, columns: Int) = FloatVectorFactory.zeroMatrix(rows, columns)
}