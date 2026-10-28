plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

repositories {
    mavenCentral()
    maven("https://central.sonatype.com/repository/maven-snapshots/") {
        mavenContent { snapshotsOnly() }
    }
}

kotlin { jvmToolchain(25) }

val ktorVersion = "3.6.0"

dependencies {
    implementation(project(":combo"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-server-body-limit:$ktorVersion")
    implementation("io.ktor:ktor-server-status-pages:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.slf4j:slf4j-simple:2.0.17")
    testImplementation(kotlin("test-junit5"))
    testImplementation("io.ktor:ktor-server-test-host:$ktorVersion")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.14.0")
}

val numericalJvmArgs = listOf("--enable-native-access=ALL-UNNAMED", "--add-modules=jdk.incubator.vector")

application {
    applicationName = "eignex-server"
    mainClass.set("com.eignex.server.ApplicationKt")
    applicationDefaultJvmArgs = numericalJvmArgs
}

tasks.test {
    useJUnitPlatform()
    jvmArgs(numericalJvmArgs)
}
