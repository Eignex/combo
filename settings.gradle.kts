rootProject.name = "combo"

include(":combo", ":server")

pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

// Opt in when working on the sibling libraries together; normal builds use published dependencies.
if (providers.gradleProperty("dependencies.local").isPresent || providers.gradleProperty("klause.local").isPresent) {
    includeBuild(providers.gradleProperty("klause.local").orNull?.ifBlank { "../klause" } ?: "../klause") {
        dependencySubstitution {
            for (module in listOf("klause", "klause-jvm", "klause-linuxx64", "klause-linuxarm64", "klause-macosarm64")) {
                substitute(module("com.eignex:$module")).using(project(":klause"))
            }
        }
    }
}

// Use unpublished Kumulant changes without rebuilding Klause.
if (providers.gradleProperty("dependencies.local").isPresent ||
    providers.gradleProperty("kumulant.local").isPresent
) {
    includeBuild(providers.gradleProperty("kumulant.local").orNull?.ifBlank { "../kumulant" } ?: "../kumulant") {
        dependencySubstitution {
            for (module in listOf("kumulant", "kumulant-jvm", "kumulant-linuxx64", "kumulant-linuxarm64", "kumulant-macosarm64")) {
                substitute(module("com.eignex:$module")).using(project(":kumulant"))
            }
        }
    }
}

// Resolve general numerical primitives from a sibling checkout during cross-library development.
providers.gradleProperty("koblas.local").orNull?.let { requested ->
    includeBuild(requested.ifBlank { "../koblas" }) {
        dependencySubstitution {
            for (module in listOf("koblas", "koblas-jvm", "koblas-linuxx64", "koblas-linuxarm64", "koblas-macosarm64")) {
                substitute(module("com.eignex:$module")).using(project(":koblas"))
            }
        }
    }
}
