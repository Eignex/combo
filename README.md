# COMBO
Combo is a library for Constraint Oriented Multi-variate Bandit Optimization (COMBO) applied to software parameters. It is used to optimize software with user data in a production environment. It supports multiple methods with a combination of machine learning, combinatorial optimization, and Thompson sampling. Some of the supported ML algorithms are: generalized linear model (GLM), random forest, deep learning, and genetic algorithms. Using COMBO, each user recieve their own configuration with potentially thousands of variables in milliseconds. As the results of each users experience with their configuration is recorded the resulting configurations will be better and better. Depending on the method employed this can require some statistical modeling. Combo is written in Kotlin with Java/JavaScript interoperability in mind.

Using it requires three steps: 

1. Create a model of the variables and constraints in the search space.
2. Map the model to your actual software features.
3. Create an multi-variate multi-armed bandit algorithm optimizer.

## Model of the search space

A model describes the variables in the optimization problem in a tree structure. Lets start of with a simple example, which is intended to be used to display a top-list of the most important media categories on a web site. Here, the optimal configuration will be automatically calculated over time as users are using it, based on how well each category performs in terms of eg sales or click data.

```kotlin
fun main() {

    val myModel = model {

        // Context variables
        int("DisplayWidth", min = 640, max = 1920)
        val customerType = nominal("CustomerType", "Child", "Company", "Person")

        // The category tree is encoded directly

        //Games can be absent
        val games = optionalMultiple("Games", "Shooter", "Platform", "Sports", "Action", "Adventure", "Strategy")

        // Movies is a sub-category with multiple sub-options
        val movies = model("Movies") {
            val horror = optionalMultiple("Horror", "Slasher", "Splatter", "Zombie")
            optionalMultiple("Action", "Thriller", "Martial arts", "Crime")
            optionalMultiple("Sci-fi", "Supernatural", "Super heroes", "Fantasy")

            // This adds a constraint that ensures that whenever CustomerType is Child then any of the 
            // horror categories are hidden.
            val child = customerType.option("Child")
            impose { child equivalent !horror }
        }

        // ... add more categories as needed

        // Add a hard constraint that the number of categories must be between 2 and 5.
        // This could have been a dynamic parameter with the number of movie genres instead.
        val categoryVariables = games.values + movies.scope.variables.asSequence()
                .flatMap { (it as Multiple<*>).values }.toList().toTypedArray()
        impose { atLeast(2, *categoryVariables) }
        impose { atMost(5, *categoryVariables) }
    }
}
```

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
