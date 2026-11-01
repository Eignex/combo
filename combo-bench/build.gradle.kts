plugins {
    kotlin("jvm")
    application
}

repositories {
    mavenCentral()
    maven("https://central.sonatype.com/repository/maven-snapshots/") {
        mavenContent { snapshotsOnly() }
    }
    exclusiveContent {
        forRepository {
            ivy {
                name = "labBenchReleases"
                url = uri(providers.gradleProperty("labBenchRepository")
                    .getOrElse("https://github.com/Eignex/eignex-lab/releases/download"))
                patternLayout { artifact("lab-bench-v[revision]/[artifact]-[revision](-[classifier]).[ext]") }
                metadataSources { gradleMetadata() }
            }
        }
        filter { includeModule("com.eignex", "lab-bench") }
    }
}

kotlin { jvmToolchain(25) }

dependencies {
    implementation(project(":combo"))
    implementation("com.eignex:lab-bench:${providers.gradleProperty("labBenchVersion").getOrElse("0.1.0")}")
    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.14.0")
}

val numericalJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")

application {
    mainClass.set("com.eignex.combo.bench.LabAdapterKt")
    applicationDefaultJvmArgs = listOf("-Xmx1g") + numericalJvmArgs
}

tasks.test {
    useJUnitPlatform()
    jvmArgs(numericalJvmArgs)
}
