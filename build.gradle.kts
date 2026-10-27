plugins {
    base
    id("com.eignex.kmp") version "1.3.5" apply false
    kotlin("plugin.serialization") version "2.4.21" apply false
}

tasks.check { dependsOn(":combo:check", ":server:check") }
tasks.assemble { dependsOn(":combo:assemble", ":server:assemble") }

tasks.register("lintDocs") {
    group = "verification"
    description = "Checks library documentation and style."
    dependsOn(":combo:lintDocs")
}
