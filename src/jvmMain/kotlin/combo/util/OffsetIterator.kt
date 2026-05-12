package combo.util

class OffsetIterator(val offset: Int, val base: IntIterator) : IntIterator() {
    override fun hasNext() = base.hasNext()
    override fun nextInt() = base.nextInt() + offset
}
