plugins {
    id("com.eignex.kmp") version "1.3.3"
    kotlin("plugin.serialization") version "2.4.10"
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
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

kotlin {
    jvm()
    // Match the targets available in Klause's published snapshots.
    linuxX64()

    sourceSets {
        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.11.0")
        }
        // Pin platform artifacts directly: KMP snapshot roots redirect to mutable
        // platform versions and lose dependency exclusions across that redirect.
        for ((sourceSet, modules) in mapOf(
            "jvmMain" to mapOf(
                "klause-jvm" to "0.0.1-20261023.210726-928",
                "kumulant-jvm" to "0.3.4-20261023.215019-68",
                "koblas-jvm" to "0.1.1-20261023.051622-267",
            ),
            "linuxX64Main" to mapOf(
                "klause-linuxx64" to "0.0.1-20261023.210726-927",
                "kumulant-linuxx64" to "0.3.4-20261023.215019-68",
                "koblas-linuxx64" to "0.1.1-20261023.051622-263",
            ),
        )) {
            getByName(sourceSet).dependencies {
                for ((module, pinned) in modules) {
                    api("com.eignex:$module:$pinned") {
                        if (!module.startsWith("koblas-")) {
                            // Kumulant and Klause pin incompatible backend versions.
                            // Export exclusions and supply the shared Koblas above.
                            exclude(group = "com.eignex", module = "koblas")
                            exclude(group = "com.eignex", module = "koblas-jvm")
                            exclude(group = "com.eignex", module = "koblas-linuxx64")
                            if (module.startsWith("klause-")) {
                                exclude(group = "com.eignex", module = "kumulant")
                                exclude(group = "com.eignex", module = "kumulant-jvm")
                                exclude(group = "com.eignex", module = "kumulant-linuxx64")
                            }
                        }
                    }
                }
            }
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
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
