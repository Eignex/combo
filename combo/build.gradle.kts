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

tasks.withType<JavaExec>().configureEach { workingDir(rootDir) }

kotlin {
    jvm()
    // Match the targets available in Klause's published snapshots.
    linuxX64()

    sourceSets {
        commonMain.dependencies {
            compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-core:1.12.0")
        }
        // Pin platform artifacts directly: KMP snapshot roots redirect to mutable
        // platform versions and lose dependency exclusions across that redirect.
        for ((sourceSet, modules) in mapOf(
            "jvmMain" to mapOf(
                "klause-jvm" to "0.0.1-20261028.153451-1056",
                "kumulant-jvm" to "0.3.4-20261027.205951-103",
            ),
            "linuxX64Main" to mapOf(
                "klause-linuxx64" to "0.0.1-20261028.153451-1055",
                "kumulant-linuxx64" to "0.3.4-20261027.205951-103",
            ),
        )) {
            getByName(sourceSet).dependencies {
                for ((module, pinned) in modules) {
                    api("com.eignex:$module:$pinned") {
                        if (!module.startsWith("koblas-")) {
                            // Kumulant and Klause pin different numerical and codec versions.
                            // Export exclusions and supply the shared platform artifacts below.
                            exclude(group = "com.eignex", module = "koblas")
                            exclude(group = "com.eignex", module = "koblas-jvm")
                            exclude(group = "com.eignex", module = "koblas-linuxx64")
                            exclude(group = "com.eignex", module = "skema")
                            exclude(group = "com.eignex", module = "skema-jvm")
                            exclude(group = "com.eignex", module = "skema-linuxx64")
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
            api("com.eignex:koblas-jvm:0.1.1-20261028.120512-279")
            api("com.eignex:skema-jvm:0.3.1-20261028.100806-19")
            api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.12.0")
        }
        linuxX64Main.dependencies {
            api("com.eignex:koblas-linuxx64:0.1.1-20261028.120512-275")
            api("com.eignex:skema-linuxx64:0.3.1-20261028.100806-19")
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
            implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.12.0")
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

// Score exported SMAC predictions with the same statistical primitives as Combo diagnostics.
tasks.register<JavaExec>("scoreSmacFrozen") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.SmacFrozenScoringKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Noisy recommendation/model/search diagnostics with separately budgeted fresh confirmation.
tasks.register<JavaExec>("refineNoisyBo") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.NoisyBoBenchmarkKt")
    args(providers.gradleProperty("noisy.seeds").orElse("").get(),
        providers.gradleProperty("noisy.design").orElse("measurement-aware").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Refit archived failing histories to separate model error from recommendation-search error.
tasks.register<JavaExec>("auditLegacyNoisyBo") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.LegacyNoisyAuditKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Optional paired comparison of joint acquisitions and existing sequential batch policies.
tasks.register<JavaExec>("compareJointBatch") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.JointBatchBenchmarkKt")
    args(providers.gradleProperty("joint.seeds").orElse("").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Export frozen GP acquisition fixtures for the independent BoTorch reference.
tasks.register<JavaExec>("auditGpAcquisition") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpAcquisitionAuditKt")
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Optional GP acquisition/search refinement suite; truth is used only for diagnostics.
tasks.register<JavaExec>("refineGp") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpRefinementBenchmarkKt")
    args(providers.gradleProperty("gpRefinement.mode").orElse("validation").get(),
        providers.gradleProperty("gpRefinement.seeds").orElse("").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Whole-round cached-posterior benchmark; full-factor reference shares the same moments and normals.
tasks.register<JavaExec>("profileGpFactorCache") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpFactorCacheBenchmarkKt")
    args(providers.gradleProperty("followup.fork").orElse("0").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")
}

// Frozen recommendation diagnosis, sample convergence, and allocation measurements.
tasks.register<JavaExec>("auditGpFollowups") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpFollowupBenchmarkKt")
    args(providers.gradleProperty("followup.mode").orElse("recommendation").get(),
        providers.gradleProperty("followup.fork").orElse("0").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector", "-Xmx1g")
    providers.gradleProperty("followup.jfr").orNull?.let { recording ->
        jvmArgs("-XX:StartFlightRecording=filename=$recording,settings=profile")
    }
}

// Paired frozen and fresh whole-loop comparison of normal base-sample policies.
tasks.register<JavaExec>("refineGpSampling") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpSamplingBenchmarkKt")
    args(providers.gradleProperty("qmc.mode").orElse("frozen").get(),
        providers.gradleProperty("qmc.seeds").orElse("").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector", "-Xmx1g")
}

// Controlled frozen-history fitting ablations and independent whole-loop confirmation.
tasks.register<JavaExec>("refineGpFitting") {
    dependsOn("jvmTestClasses")
    val compilation = kotlin.targets.getByName("jvm").compilations.getByName("test")
    classpath(compilation.output.allOutputs, compilation.runtimeDependencyFiles)
    mainClass.set("com.eignex.combo.bo.GpFittingBenchmarkKt")
    args(providers.gradleProperty("fitting.mode").orElse("frozen").get(),
        providers.gradleProperty("fitting.variants").orElse("baseline24,baseline96,combined96").get(),
        providers.gradleProperty("fitting.seeds").orElse("1901,1902,1903,1904,1905,1906").get())
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(25)) })
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector", "-Xmx1g")
}
