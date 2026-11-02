import java.net.URI
import java.security.MessageDigest

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

// Historical measurements are inputs to a frozen-history audit, never checked-in result archives.
tasks.register("prepareDiagnosticFixtures") {
    val destination = layout.buildDirectory.file("fixtures/gp-fitting-validation.csv.gz")
    val expected = "e51cc99f441a616c1f24272555026ec93383e6aef1fef118fde6c2bf28e29d09"
    outputs.file(destination)
    doLast {
        val file = destination.get().asFile
        fun digest(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it) }
        if (file.isFile && digest(file.readBytes()) == expected) return@doLast
        val url = URI("https://raw.githubusercontent.com/Eignex/combo/" +
            "3b1b1316c962ea0fd2593482f0ceb79b9aad1fac/benchmarks/gp-fitting/validation-traces.csv.gz").toURL()
        val bytes = url.openConnection().apply { connectTimeout = 30_000; readTimeout = 60_000 }.getInputStream().use { it.readBytes() }
        require(digest(bytes) == expected) { "Historical fixture checksum mismatch" }
        file.parentFile.mkdirs()
        file.writeBytes(bytes)
    }
}
