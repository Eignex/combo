package com.eignex.combo.bandit.dt

import com.eignex.combo.bandit.PredictionBanditTestSuite
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.shared.PredictionLearner
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import com.eignex.kumulant.stat.summary.MeanStat

class HybridRandomForestSuiteTest : PredictionBanditTestSuite<ForestData>() {
    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionLearner<ForestData> = RandomForestBandit.build(
        space = space,
        nbrTrees = 5,
        mtry = 100,
        config = RegressionTreeConfig(
            splitPeriod = 5,
            minSamplesSplit = 10.0,
            minSamplesLeaf = 2.0,
            delta = 0.1,
        ),
        candidateBatchSize = 4,
        randomSeed = randomSeed,
        maximize = maximize,
        rewards = rewards,
    )
}
