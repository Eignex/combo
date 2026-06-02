@file:OptIn(ExperimentalWasmDsl::class)

import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    id("com.eignex.kmp") version "1.2.2"
    kotlin("plugin.serialization") version "2.3.20"
}

eignexPublish {
    description.set("Multi-armed bandit algorithms for combinatorial decision spaces.")
    githubRepo.set("Eignex/combo")
}

kotlin {
    jvm()
    js(IR) { browser(); nodejs() }
    // wasmJs is disabled to match our klause api dependency, which disables it due to a
    // Kotlin/Wasm 2.3.20 miscompile of its propagation fixpoint loop. kbuild 1.2.2's
    // partial-resolution checker requires every commonMain dependency to cover all of our
    // targets, so combo cannot target wasmJs while klause omits it. Re-enable together.
    // wasmJs { browser(); nodejs() }
    wasmWasi { nodejs() }
    linuxX64(); linuxArm64()
    macosX64(); macosArm64(); mingwX64()
    iosX64(); iosArm64(); iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            api("com.eignex:klause")
            implementation("com.eignex:kumulant:0.3.1")
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.10.0")
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.10.0")
        }
    }
}


dokka {
    dokkaSourceSets.configureEach {
        sourceLink {
            localDirectory.set(projectDir.resolve("src"))
            val sub = projectDir.relativeTo(rootDir).invariantSeparatorsPath
            val prefix = if (sub.isEmpty()) "src" else "$sub/src"
            remoteUrl("https://github.com/Eignex/${rootProject.name}/blob/main/$prefix")
            remoteLineSuffix.set("#L")
        }
    }
}
