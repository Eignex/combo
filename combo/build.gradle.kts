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
            api("com.eignex:klause:0.0.1-SNAPSHOT") {
                exclude(group = "com.eignex", module = "kumulant")
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:kumulant:0.3.4-20261027.205951-103") {
                version { strictly("0.3.4-20261027.205951-103") }
                exclude(group = "com.eignex", module = "koblas")
                exclude(group = "com.eignex", module = "skema")
            }
            api("com.eignex:koblas:0.1.1-20261028.120512-276") {
                version { strictly("0.1.1-20261028.120512-276") }
            }
            api("com.eignex:skema:0.3.1-20261028.100806-19") {
                version { strictly("0.3.1-20261028.100806-19") }
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
// downstream consumers use the same numerical and codec binaries as this build.
dependencies {
    constraints {
        for ((library, platforms) in mapOf(
            "kumulant" to mapOf(
                "jvm" to "0.3.4-20261027.205951-103",
                "linuxx64" to "0.3.4-20261027.205951-103",
                "linuxarm64" to "0.3.4-20261027.205951-103",
                "macosarm64" to "0.3.4-20261027.205951-102",
            ),
            "koblas" to mapOf(
                "jvm" to "0.1.1-20261028.120512-279",
                "linuxx64" to "0.1.1-20261028.120512-275",
                "linuxarm64" to "0.1.1-20261028.120512-276",
                "macosarm64" to "0.1.1-20261028.120512-299",
            ),
            "skema" to mapOf(
                "jvm" to "0.3.1-20261028.100806-19",
                "linuxx64" to "0.3.1-20261028.100806-19",
                "linuxarm64" to "0.3.1-20261028.100806-19",
                "macosarm64" to "0.3.1-20261028.100806-19",
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
