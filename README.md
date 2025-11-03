## This project is currently under transformation into a cloud-native self hosted tool.

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](LICENSE)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.2%2B-orange)](https://kotlinlang.org/)
<!--[![Build](https://github.com/Eigenity/combo/actions/workflows/build.yml/badge.svg)](https://github.com/Eigenity/combo/actions/workflows/build.yml)-->
<!-- [![Docker](https://img.shields.io/badge/Docker-ghcr.io%2FEigenity%2Fcombo-blue)](https://ghcr.io/Eigenity/combo) -->
<!-- [![Helm](https://img.shields.io/badge/Helm-chart-blueviolet)](https://artifacthub.io/packages/helm/eigenity/combo) -->

> **Kubernetes-ready Kotlin tool for optimizing LLM prompts, UIs, or other software parameters from real user feedback.**

---

## 🚀 Overview

**COMBO** (Constraint Oriented Multi-variate Bandit Optimization) is a **cloud-native, self-hosted optimization engine** for adaptive systems.  It continuously improves **software configurations** using **real interaction data**. COMBO can optimize thousands of variables per user in milliseconds, automatically learning what works best as feedback accumulates.

## Installation
- **As a JVM library** using Kotlin or Java using
`gradle todo`
- **As a Self-hosted Web Server**. Deploy easily via Helm to self-hosted Kubernetes clusters (WIP)

## Key features 
- **LLM & UI Optimization**: Tune prompts, layouts, or behaviors based on user outcomes
- **Declarative Search Spaces**: Define structured variable models with constraints through SDKs or HTTP API
- **Personalized Configurations**: Generate and refine per-user configurations in real time
- **Flexible Optimization Engine**: Easily setup your optimization engine through sane defaults and robust algorithms

---

## 🔧 Quick Start

Currently only available as a maven central library.

1. **Define the search space**
   ```kotlin
    val model = model {

        // Context: available token budget
        int("TokenBudget", min = 500, max = 4000)

        // Parameters to optimize
        val systemTone = nominal("SystemTone", "Friendly", "Professional", "Concise", "Creative")
        val includeExamples = boolean("IncludeExamples")
        val verbosity = int("VerbosityLevel", min = 1, max = 5)

        // Estimated token cost for each element
        val toneCost = mapOf(
            "Friendly" to 200,
            "Professional" to 250,
            "Concise" to 150,
            "Creative" to 300
        )

        // Impose a linear constraint on total token usage
        impose {
            // approximate token usage = base + toneCost + verbosity*100 + examples*500
            val totalTokens = verbosity * 100 +
                (includeExamples.asInt() * 500) +
                systemTone.mapValues(toneCost).sum() + 200 // base system cost

            totalTokens lessThanEq getInt("TokenBudget")
        }
    }
    ```

2. **Create and use an optimizer**

    ```kotlin
    val optimizer = RandomForestBandit.Builder(model)
    val assignment = optimizer.chooseOrThrow()
    println("Prompt config: ${assignment.toMap()}")

    // Simulate user feedback (e.g., rating or completion quality)
    optimizer.update(assignment, 0.85f)
    ```
