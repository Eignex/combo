plugins {
    kotlin("jvm") version "2.2.0"
}

group = "egenity.combo"
version = "0.2-SNAPSHOT"

repositories {
    mavenCentral()
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

kotlin {
    jvmToolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.ow2.sat4j:org.ow2.sat4j.maxsat:2.3.5")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
