package combo.decisions

/**
 * Contextual feature observed at choose-time. The value is supplied per-call via
 * [Context]; klause never sees it.
 */
data class BoolContextHandle(val name: String)

/**
 * Contextual integer feature observed at choose-time. The value is supplied per-call
 * via [Context]; klause never sees it.
 */
data class IntContextHandle(val name: String)
