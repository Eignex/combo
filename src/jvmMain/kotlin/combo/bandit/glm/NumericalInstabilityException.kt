package combo.bandit.glm

/**
 * Raised when a linear model produces a non-finite prediction or its covariance
 * estimate drifts out of the positive-definite cone. Callers should treat it as
 * a signal to reset the model (see [LinearModel.blank]) rather than recover.
 */
class NumericalInstabilityException(message: String) : RuntimeException(message)
