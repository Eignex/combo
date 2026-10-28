package com.eignex.combo.test

import org.junit.Rule
import org.junit.rules.Timeout

actual abstract class TimedTest actual constructor() {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(30)
}
