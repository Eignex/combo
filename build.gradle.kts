plugins {
    base
    id("com.eignex.kmp") apply false
    kotlin("plugin.serialization") version "2.4.21" apply false
}

allprojects {
    repositories {
        ivy {
            url = uri("https://github.com/Eignex/kbuild/releases/download")
            patternLayout { artifact("v[revision]/[artifact]-[revision](-[classifier]).[ext]") }
            metadataSources { gradleMetadata() }
            content { includeModule("com.eignex", "kbuild-platform") }
        }
    }
}

subprojects {
    configurations.configureEach {
        resolutionStrategy.cacheChangingModulesFor(0, "seconds")
    }
}

tasks.check { dependsOn(":combo:check", ":combo-bench:check", ":server:check") }
tasks.assemble { dependsOn(":combo:assemble", ":server:assemble") }

tasks.register("lintDocs") {
    group = "verification"
    description = "Checks library documentation and style."
    dependsOn(":combo:lintDocs")
}
