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
                "kumulant-jvm" to "0.3.4-20261025.053320-73",
            ),
            "linuxX64Main" to mapOf(
                "klause-linuxx64" to "0.0.1-20261023.210726-927",
                "kumulant-linuxx64" to "0.3.4-20261025.053320-73",
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
        // Public numerical APIs use Koblas directly, independently of the statistical libraries.
        jvmMain.dependencies {
            api("com.eignex:koblas-jvm:0.1.1-20261024.213917-268")
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
        linuxX64Main.dependencies { api("com.eignex:koblas-linuxx64:0.1.1-20261024.213917-264") }
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

// Optional external comparison; run scripts/run_smac_reference.py first.
tasks.register<JavaExec>("compareSmac") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.SmacComparisonKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Optional frozen-history and whole-loop refinement experiments.
tasks.register<JavaExec>("refineBo") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.RefinementBenchmarkKt")
    args(providers.gradleProperty("refinement.mode").orElse("all").get(),
        providers.gradleProperty("refinement.seeds").orElse("").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Optional GP/forest/random comparison over shared frozen histories and equal external budgets.
tasks.register<JavaExec>("compareGp") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GaussianProcessBenchmarkKt")
    args(providers.gradleProperty("gp.mode").orElse("all").get(),
        providers.gradleProperty("gp.seeds").orElse("").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}
