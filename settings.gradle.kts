rootProject.name = "combo"

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

// Opt in when working on the sibling libraries together; normal builds use the pinned artifacts.
if (providers.gradleProperty("dependencies.local").isPresent) {
    includeBuild("../klause") {
        dependencySubstitution {
            substitute(module("com.eignex:klause-jvm")).using(project(":klause"))
            substitute(module("com.eignex:klause-linuxx64")).using(project(":klause"))
        }
    }
}

// Use unpublished Kumulant changes without rebuilding Klause.
if (providers.gradleProperty("dependencies.local").isPresent ||
    providers.gradleProperty("kumulant.local").isPresent
) {
    includeBuild("../kumulant") {
        dependencySubstitution {
            substitute(module("com.eignex:kumulant-jvm")).using(project(":kumulant"))
            substitute(module("com.eignex:kumulant-linuxx64")).using(project(":kumulant"))
        }
    }
}
