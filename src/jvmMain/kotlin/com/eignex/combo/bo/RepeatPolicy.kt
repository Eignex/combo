package com.eignex.combo.bo

/** Whether completed original-coordinate assignments may consume another external evaluation. */
enum class RepeatPolicy {
    /** Allow replicates only when the outcome is explicitly declared noisy. */
    NoisyOnly,

    /** Allow replicates, including compatibility choose/update loops over exhausted finite spaces. */
    Allow,

    /** Exclude completed assignments even when additional noisy measurements could be useful. */
    Never,
}
