plugins {
    kotlin("jvm") version "2.0.21"
    //`maven-publish`
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

kotlin {
    jvmToolchain(21)
}

dependencies {
    compileOnly("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    compileOnly("org.ow2.sat4j:org.ow2.sat4j.maxsat:2.3.5")

    // Test dependencies
    testImplementation(kotlin("test"))
    testImplementation(kotlin("test-junit5"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.1")
    testImplementation("org.deeplearning4j:deeplearning4j-nn:1.0.0-beta7")
    testImplementation("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    testImplementation("org.ow2.sat4j:org.ow2.sat4j.maxsat:2.3.5")
    testImplementation("org.jacop:jacop:4.7.0")
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        freeCompilerArgs = listOf("-Xjvm-default=all")
        jvmTarget = "21"
    }
}

tasks.wrapper {
    gradleVersion = "8.5"
    distributionType = Wrapper.DistributionType.ALL
}

/*
publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])

            pom {
                name.set("COMBO")
                description.set("Constraint Oriented Multi-variate Bandit Optimization library")
                url.set("https://github.com/rasros/combo")

                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
            }
        }
    }
}
 */
