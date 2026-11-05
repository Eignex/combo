plugins {
    id("com.eignex.kmp")
    kotlin("plugin.serialization")
}

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
            api("com.eignex:klause:0.0.1-20261105.183337-1210") {
                version { strictly("0.0.1-20261105.183337-1210") }
                exclude(group = "com.eignex", module = "kumulant")
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:kumulant:0.3.4-20261105.142345-130") {
                version { strictly("0.3.4-20261105.142345-130") }
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:koblas:0.1.1-20261105.140327-312") {
                version { strictly("0.1.1-20261105.140327-312") }
            }
            api("com.eignex:skema:0.4.0")
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
                "jvm" to "0.0.1-20261105.183337-1202",
                "linuxx64" to "0.0.1-20261105.183337-1201",
            ),
            "kumulant" to mapOf(
                "jvm" to "0.3.4-20261105.142345-130",
                "linuxx64" to "0.3.4-20261105.142345-130",
                "linuxarm64" to "0.3.4-20261105.142345-130",
                "macosarm64" to "0.3.4-20261105.142345-129",
            ),
            "koblas" to mapOf(
                "jvm" to "0.1.1-20261105.140327-315",
                "linuxx64" to "0.1.1-20261105.140327-310",
                "linuxarm64" to "0.1.1-20261105.140327-312",
                "macosarm64" to "0.1.1-20261105.140327-334",
            ),
        )) {
            for ((platform, pinned) in platforms) {
                add("commonMainApi", "com.eignex:$library-$platform:$pinned") {
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
