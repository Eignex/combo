package com.eignex.combo.decisions

private val context = ThreadLocal<SubSpaceContext?>()

internal actual var currentSubSpaceContext: SubSpaceContext?
    get() = context.get()
    set(value) = context.set(value)
