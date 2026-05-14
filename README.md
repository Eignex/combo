<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

This project is currently under transformation into a cloud-native self hosted tool.

# COMBO
Combo is a library for Constraint Oriented Multi-variate Bandit Optimization (COMBO) applied to software parameters. It is used to optimize software with user data in a production environment. It supports multiple methods with a combination of machine learning, combinatorial optimization, and Thompson sampling. Some of the supported ML algorithms are: generalized linear model (GLM), random forest, deep learning, and genetic algorithms. Using COMBO, each user recieve their own configuration with potentially thousands of variables in milliseconds. As the results of each users experience with their configuration is recorded the resulting configurations will be better and better. Depending on the method employed this can require some statistical modeling.

Using it requires three steps: 

1. Create a model of the variables and constraints in the search space.
2. Map the model to actual code behavior.
3. Create a multi-variate multi-armed bandit algorithm optimizer.

## Model of the search space

A model describes the variables in the optimization problem in a tree structure. Lets start of with a simple example, which is intended to be used to display a top-list of the most important media categories on a web site. Here, the optimal configuration will be automatically calculated over time as users are using it, based on how well each category performs in terms of eg sales or click data.

```json
{
  "decisionSpace": {
    "context": {
      "displayWidth": { "type": "int", "min": 640, "max": 1920 },
      "customerType": { "type": "nominal", "labels": ["Child", "Company", "Person"] }
    },

    "multiples": {
      "games": ["Shooter", "Platform", "Sports", "Action", "Adventure", "Strategy"]
    },

    "spaces": {
      "movies": {
        "multiples": {
          "horror": ["Slasher", "Splatter", "Zombie"],
          "action": ["Thriller", "MartialArts", "Crime"],
          "sciFi":  ["Supernatural", "SuperHeroes", "Fantasy"]
        },
        "constraints": [
          "customerType == \"Child\" implies |horror| == 0"
        ]
      }
    },

    "constraints": [
      "|games| + |movies.horror| + |movies.action| + |movies.sciFi| in [2, 5]"
    ]
  }
}
```

A `multiple` is a set-valued variable: it expands at compile time to one boolean
indicator per label (`games.Shooter`, `games.Platform`, …). Inside `constraint { … }`
the matching Kotlin DSL reads `games.contains("Shooter")`, `games.containsAny(...)`,
`games.containsAll(...)`, `games.sizeLe(5)`, `games.sizeBetween(2, 5)`. Same
declaration surface as `nominal`, the only difference is "pick one" vs. "pick any
subset".

## Optimizer

Creating an optimizer is straightforward. There are several hyper-parameters that can be tuned for better performance. The random forest algorithm is recommended to start with because it is quite robust to bad tuning.

```kotlin
// Using the feature model "myModel" from above
// This optimizer will maximize binomial data (success/failures).
val optimizer = RandomForestBandit.Builder(myModel)
```

Using the optimizer then is as simple as this:

```kotlin
val assignment1 = optimizer.chooseOrThrow()
// The values can be queried like so:
assignment1.getBoolean("Horror")
// It can be used as an ordinary map from String to value as such (but then the structure is lost).
val map = assignment1.toMap()

// Update with the result of an assignment
optimizer.update(assignment1, 1f)
```

To get a "personalized" assignment do this:

```kotlin
val assignment2 = optimizer.chooseOrThrow(myModel["DisplayWidth", 1920], myModel["CustomerType", "Child"])
optimizer.update(assignment2, 0f)
```
