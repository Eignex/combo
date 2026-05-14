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
            implementation("com.eignex:kumulant:0.1.1")
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
val pendingRewire = listOf<String>(
    // DT slice 1 rewire complete. RandomForestBandit and serialisable TreeData
    // round-trip land in follow-up slices; their files were deleted rather than
    // excluded so the package compiles cleanly without dead code.
)
val pendingRewireTests = listOf<String>()
if (pendingRewire.isNotEmpty()) kotlin.sourceSets["jvmMain"].kotlin.exclude(pendingRewire)
if (pendingRewireTests.isNotEmpty()) kotlin.sourceSets["jvmTest"].kotlin.exclude(pendingRewireTests)
