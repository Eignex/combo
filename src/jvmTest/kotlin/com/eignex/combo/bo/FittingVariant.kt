package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessFitting

internal data class FittingVariant(
    val name: String,
    val trials: Int = 96,
    val fitting: GaussianProcessFitting = GaussianProcessFitting(),
)
