package combo.bandit

enum class ParallelMode {

    /**
     * Updates are queued non-blocking through a linked list with compare-and-swap. The update queue can grow
     * indefinitely. Processing updates cannot block and batch min-size cannot be guaranteed.
     */
    NON_LOCKING,

    /**
     * Updates are queued synchronized with a read/write lock. The update queue can grow indefinitely. Processing
     * updates can block if requested and batch min-size is guaranteed.
     */
    LOCKING,

    /**
     * For cases where the updates must not lag too far behind. This is intended to be used in simulations.
     * When there is more than maximum batch size unprocessed input events (update/updateAll calls) the choose
     * method will block until the data has been processed.
     */
    BLOCKING,
}