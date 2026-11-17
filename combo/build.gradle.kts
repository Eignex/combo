plugins {
    id("com.eignex.kmp")
    kotlin("plugin.serialization")
}

eignexBuild { jvmToolchain.set(25) }

eignexPublish {
    description.set("Streaming bandits and history-based Bayesian optimization over constrained decision spaces.")
    githubRepo.set("Eignex/combo")
}

repositories {
    maven("https://central.sonatype.com/repository/maven-snapshots/") {
        mavenContent { snapshotsOnly() }
    }
}

// Record existing findings from Kbuild's newer documentation and style checks.
detekt {
    baseline.set(file("detekt-baseline.xml"))
}

tasks.withType<Test>().configureEach {
    workingDir(rootDir)
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

val klausePin = "0.0.1-20261117.021241-1352"
val kumulantPin = "0.3.4-20261116.231003-146"
val koblasPin = "0.1.1-20261116.224320-351"
val skemaPin = "0.4.1-20261116.152019-3"
val utilPin = "0.0.1-ci.43969094073.1-20261116.222329-1"
val kpermutePin = "1.3.1-20261116.152032-8"

// Match Klause's default test targets; releases include both ARM64 compute hosts.
val fullTargets = providers.gradleProperty("targets.full").isPresent

@OptIn(org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation::class)
kotlin {
    // Conditional target sets need matching ABI references in both build modes.
    abiValidation {
        referenceDumpDir.set(layout.projectDirectory.dir(if (fullTargets) "api-full" else "api"))
    }
    applyDefaultHierarchyTemplate()
    compilerOptions {
        freeCompilerArgs.add("-Xexpect-actual-classes")
    }
    jvm()
    linuxX64()
    if (fullTargets) {
        linuxArm64()
        macosArm64()
    }

    sourceSets {
        commonMain.dependencies {
            api("com.eignex:klause:$klausePin") {
                version { strictly(klausePin) }
                exclude(group = "com.eignex", module = "kumulant")
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:kumulant:$kumulantPin") {
                version { strictly(kumulantPin) }
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:koblas:$koblasPin") {
                version { strictly(koblasPin) }
            }
            api("com.eignex:skema:$skemaPin") {
                version { strictly(skemaPin) }
            }
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.12.0")
            api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.12.0")
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.12.0")
        }
    }
}

// A timestamped KMP root redirects to mutable platform snapshots. Export constraints so
// downstream consumers use the same numerical binaries as this build.
dependencies {
    constraints {
        for ((library, platforms) in mapOf(
            "klause" to mapOf(
                "" to klausePin,
                "jvm" to "0.0.1-20261117.021241-1347",
                "linuxx64" to "0.0.1-20261117.021241-1345",
                "linuxarm64" to "0.0.1-20261117.021241-3",
                "macosarm64" to "0.0.1-20261117.021241-3",
            ),
            "kumulant" to mapOf(
                "" to kumulantPin,
                "jvm" to kumulantPin,
                "linuxx64" to kumulantPin,
                "linuxarm64" to kumulantPin,
                "macosarm64" to "0.3.4-20261116.231003-145",
            ),
            "koblas" to mapOf(
                "" to koblasPin,
                "jvm" to "0.1.1-20261116.224320-354",
                "linuxx64" to "0.1.1-20261116.224320-349",
                "linuxarm64" to koblasPin,
                "macosarm64" to "0.1.1-20261116.224320-373",
            ),
            "skema" to mapOf(
                "" to skemaPin,
                "jvm" to skemaPin,
                "linuxx64" to skemaPin,
                "linuxarm64" to skemaPin,
                "macosarm64" to skemaPin,
            ),
            "util" to mapOf(
                "" to utilPin,
                "jvm" to utilPin,
                "linuxx64" to utilPin,
                "linuxarm64" to utilPin,
                "macosarm64" to utilPin,
            ),
            "kpermute" to mapOf(
                "" to kpermutePin,
                "jvm" to kpermutePin,
                "linuxx64" to kpermutePin,
                "linuxarm64" to kpermutePin,
                "macosarm64" to kpermutePin,
            ),
        )) {
            for ((platform, pinned) in platforms) {
                val module = if (platform.isEmpty()) library else "$library-$platform"
                add("commonMainApi", "com.eignex:$module:$pinned") {
                    version { strictly(pinned) }
                }
            }
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
