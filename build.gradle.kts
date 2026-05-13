@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    id("com.eignex.kmp") version "1.1.4"
    kotlin("plugin.serialization") version "2.3.0"
}

eignexPublish {
    description.set("Multi-armed bandit algorithms for combinatorial decision spaces.")
    githubRepo.set("Eignex/combo")
}

kotlin {
    jvm()
    js(IR) { browser(); nodejs() }
    wasmJs { browser(); nodejs() }
    wasmWasi { nodejs() }
    linuxX64(); linuxArm64()
    macosX64(); macosArm64(); mingwX64()
    iosX64(); iosArm64(); iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            api("com.eignex:klause")
            implementation("com.eignex:kumulant:0.1.0")
            implementation("com.eignex:kpermute:1.1.2")
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.10.0")
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.10.0")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.10.0")
        }
        jvmMain.dependencies {
            api("com.eignex:klause-logicng")
            implementation("org.slf4j:slf4j-api:2.0.17")
        }
    }
}

// Files awaiting rewire onto the new bandit harness (klause Sample + kumulant Result).
// Each entry will be lifted out as its subsystem is migrated.
val pendingRewire = listOf(
    "combo/bandit/InstancesData.kt",
    "combo/bandit/ListBandit.kt",
    "combo/bandit/ParallelBandit.kt",
    "combo/bandit/UpdateEvents.kt",
    "combo/bandit/univariate/ParallelUnivariateBandit.kt",
    "combo/bandit/dt/**",
)
val pendingRewireTests = listOf(
    "combo/bandit/BanditsTest.kt",
    "combo/bandit/ListBanditTest.kt",
    "combo/bandit/univariate/BanditPoliciesTest.kt",
    "combo/bandit/univariate/ParallelUnivariateBanditTest.kt",
    "combo/bandit/univariate/UnivariatePosteriorsTest.kt",
    "combo/bandit/dt/**",
    // Depends on combo.sat.BitArray (dropped) for vector indexing — the cholesky
    // downdate path is exercised by combo.bandit.glm.LinearModelTest via
    // CovarianceLinearModel.
    "combo/math/CholeskyTest.kt",
)
kotlin.sourceSets["jvmMain"].kotlin.exclude(pendingRewire)
kotlin.sourceSets["jvmTest"].kotlin.exclude(pendingRewireTests)
