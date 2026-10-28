package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.combo.test.TimedTest
import com.eignex.klause.schema.not
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JointRefinementTest : TimedTest() {
    @Test
    fun `cached EI and log EI match frozen BoTorch values with identical normals`() {
        for ((directory, sampling) in listOf(
            "gp-followups" to JointNormalSampling.Antithetic,
            "gp-qmc" to JointNormalSampling.ScrambledSobol,
        )) {
            val fixtures = File("benchmarks/$directory/frozen-q4.json.gz").inputStream().use {
                Json.parseToJsonElement(GZIPInputStream(it).bufferedReader().readText()).jsonArray
            }
            val reference = File("benchmarks/$directory/reference-q4.csv").readLines().drop(1).associate { line ->
                val row = line.split(',')
                (row[0] to row[2]) to row[6].toDouble()
            }
            for (entry in fixtures.take(1)) {
                val fixture = entry.jsonObject
                val means = fixture.getValue("means").jsonArray.map { it.jsonPrimitive.double }
                val covariance = fixture.getValue(
                    "covariance",
                ).jsonArray.map { it.jsonArray.map { x -> x.jsonPrimitive.double } }
                val model = MatrixPosterior(means, covariance)
                val size = fixture.getValue("baseline").jsonPrimitive.int
                val baseline = jointPoints(*(0 until size).map(Int::toDouble).toDoubleArray())
                val noisy = fixture.getValue("id").jsonPrimitive.content.endsWith("qnei")
                val context = JointAcquisitionContext(
                    model,
                    fixture.getValue("incumbent").jsonPrimitive.double,
                    OutcomeSpec(),
                    baseline,
                )
                val raw = (
                    if (noisy) {
                        JointNoisyExpectedImprovement(
                            sampling,
                        )
                    } else {
                        JointExpectedImprovement(sampling)
                    }
                    ).create(
                    context,
                    Random(713),
                )
                val log = (
                    if (noisy) {
                        JointLogNoisyExpectedImprovement(
                            sampling,
                        )
                    } else {
                        JointLogExpectedImprovement(sampling)
                    }
                    ).create(
                    context,
                    Random(713),
                )
                for (case in fixture.getValue("sets").jsonArray.take(1)) {
                    val record = case.jsonObject
                    val indices = record.getValue("indices").jsonArray.map { it.jsonPrimitive.int }
                    val points = jointPoints(*indices.map(Int::toDouble).toDoubleArray())
                    assertEquals(record.getValue("legacy").jsonPrimitive.double, raw.evaluate(points), 1e-9)
                    assertEquals(
                        reference.getValue(fixture.getValue("id").jsonPrimitive.content to indices.joinToString(";")),
                        log.evaluate(points),
                        1e-8,
                    )
                }
            }
        }
    }

    @Test
    fun `log improvement remains ordered far beyond ordinary EI zero plateau`() {
        val model = object : FittedSurrogate {
            override val uncertaintyKind = UncertaintyKind.LatentPosterior
            override fun predictJoint(features: DenseMatrix) = JointPredictiveMoments(
                DenseVector.of(DoubleArray(features.rows) { features[it, 0] }),
                DenseMatrix.zero(features.rows),
            )
            override fun predict(features: DenseMatrix): PredictiveMoments {
                val joint = predictJoint(features)
                return PredictiveMoments(joint.means, DenseVector.zero(features.rows))
            }
            override fun drawFunction(random: Random): SampledFunction = error("not used")
        }
        val context = JointAcquisitionContext(model, 0.0, OutcomeSpec(), jointPoints())
        val raw = JointExpectedImprovement().create(context, Random(1))
        val log = JointLogExpectedImprovement().create(context, Random(1))
        assertEquals(0.0, raw.evaluate(jointPoints(-1.0)))
        assertEquals(0.0, raw.evaluate(jointPoints(-2.0)))
        assertTrue(log.evaluate(jointPoints(-1.0)) > log.evaluate(jointPoints(-2.0)))
        assertTrue(log.evaluate(jointPoints(-1e300)).isFinite())
        assertEquals(Double.NEGATIVE_INFINITY, log.evaluate(jointPoints()))
    }

    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }

    @Test
    fun `joint local moves improve outside the finite pool and retain feasible distinct exclusions`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(
                listOf(
                    continuousPoint(doubleArrayOf(0.0)).sample,
                    continuousPoint(doubleArrayOf(0.2)).sample,
                ),
                WitnessTermination.CountReached,
            )
        }
        val objective = JointAcquisition { rows ->
            -(0 until rows.rows).sumOf {
                (rows[it, 0] - 0.73) *
                    (rows[it, 0] - 0.73)
            }
        }
        val excluded = continuousPoint(doubleArrayOf(0.0)).sample
        val request = AcquisitionRequest(Context.Empty, listOf(excluded), listOf(excluded))
        val local = JointLocalAndRandomSearch(
            projection,
            backend,
            localSteps = 12,
            localPolicy = LocalSearchPolicy(starts = 2),
        ).maximize(objective, request, 1, Random(13))
        assertTrue(local.samples.single().sample.reals[0] > 0.65)
        assertTrue(WitnessValidator(space).valid(local.samples.single().sample))
        assertTrue(local.evaluations > 1)
        val empty = JointPoolSearch(projection, backend).maximize(
            JointAcquisition { Double.NEGATIVE_INFINITY },
            AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
            2,
            Random(1),
        )
        assertEquals(2, empty.samples.size)
        assertEquals(WitnessTermination.CountReached, empty.termination)
    }

    private class ScopedRate : DecisionSpace() {
        val premium by contextBool()
        val blocked by boolVar()
        val x by floatVar(0.0, 1.0)
        val fixed by constraint { !blocked }
    }

    @Test
    fun `joint neighborhoods preserve hard context constraints and global pending exclusions`() {
        val model = ScopedRate()
        val space = model.compileSpace()
        val projection = SearchTransform(space)
        val scope = context { set(model.premium, true) }
        val witnesses = HybridDescentSession.klause(space)
        val unavailable = witnesses.witnesses(
            com.eignex.combo.shared.WitnessRequest(1, space.assumptionsFor(scope)),
            Random(4),
        ).witnesses
        val result = JointLocalAndRandomSearch(
            projection,
            witnesses,
            candidateCount = 4,
            localSteps = 3,
            neighbors = 4,
            localPolicy = LocalSearchPolicy(starts = 2),
        ).maximize(
            JointAcquisition { rows -> (0 until rows.rows).sumOf { rows[it, rows.cols - 1] } },
            AcquisitionRequest(scope, unavailable, unavailable),
            2,
            Random(12),
        )
        assertEquals(2, result.samples.size)
        val keys = result.samples.map { it.sample.assignmentKey() }
        assertEquals(2, keys.distinct().size)
        assertTrue(keys.none { it in unavailable.map { point -> point.assignmentKey() } })
        assertTrue(
            result.samples.all {
                space.matches(
                    it.sample,
                    space.assumptionsFor(scope),
                ) && WitnessValidator(space).valid(it.sample)
            },
        )
    }

    @Test
    fun `delegating decorators retain joint overrides unless they opt into the prediction cache`() {
        val gp = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
            TrainingData(
                jointPoints(0.0),
                DenseVector.of(doubleArrayOf(1.0)),
                DenseVector.of(doubleArrayOf(1.0)),
            ),
            Random(1),
        )
        val decorated = object : FittedSurrogate by gp {
            override fun predictJoint(features: DenseMatrix): JointPredictiveMoments {
                val prediction = gp.predictJoint(features)
                return JointPredictiveMoments(
                    DenseVector.of(DoubleArray(features.rows) { prediction.means[it] + 100.0 }),
                    prediction.covariance,
                )
            }
        }
        val cached = decorated.jointPosterior(jointPoints(0.1))!!
        assertEquals(gp.predictJoint(jointPoints(0.1)).means[0] + 100, cached.baseline.means[0], 1e-12)
        assertEquals(
            gp.predictJoint(jointPoints(0.3)).means[0] + 100,
            cached.predict(jointPoints(0.3)).candidates.means[0],
            1e-12,
        )
    }

    @Test
    fun `GP bound posterior matches full covariance and captures baseline through response transforms`() {
        val gp = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
            TrainingData(
                jointPoints(0.0, 0.4, 1.0),
                DenseVector.of(doubleArrayOf(1.0, 2.0, 1.5)),
                DenseVector.of(doubleArrayOf(1.0, 1.0, 1.0)),
            ),
            Random(1),
        )
        for (model in listOf(gp, ResponseTransform(2.0, 3.0).model(gp), ResponseTransform(0.1, 0.2, true).model(gp))) {
            val baseline = jointPoints(0.1, 0.7)
            val cached = if (model is LogResponseSurrogate) {
                model.logJointPosterior(
                    baseline,
                )!!
            } else {
                model.jointPosterior(baseline)!!
            }
            baseline.values.fill(99.0)
            val prediction = cached.predict(jointPoints(0.3, 0.9))
            val all = if (model is LogResponseSurrogate) {
                model.predictLogJoint(
                    jointPoints(0.1, 0.7, 0.3, 0.9),
                )!!
            } else {
                model.predictJoint(jointPoints(0.1, 0.7, 0.3, 0.9))!!
            }
            for (i in 0..1) {
                for (j in 0..1) {
                    assertEquals(all.means[i], cached.baseline.means[i], 1e-12)
                    assertEquals(all.means[i + 2], prediction.candidates.means[i], 1e-12)
                    assertEquals(all.covariance[i, j], cached.baseline.covariance[i, j], 1e-12)
                    assertEquals(all.covariance[i + 2, j + 2], prediction.candidates.covariance[i, j], 1e-12)
                    assertEquals(all.covariance[i, j + 2], prediction.crossCovariance[i, j], 1e-12)
                }
            }
            val sampler = JointGaussianSampler(cached, 32, 7L)
            val before = sampler.baselineDraws.values.toList()
            sampler.draw(jointPoints(0.3, 0.9))
            sampler.draw(jointPoints(0.2, 0.8))
            assertEquals(1, sampler.baselineFactorizations)
            assertEquals(2, sampler.candidateFactorizations)
            assertEquals(before, sampler.baselineDraws.values.toList())
        }
    }
}
internal class MatrixPosterior(private val means: List<Double>, private val covariance: List<List<Double>>) :
    FittedSurrogate {
    override val uncertaintyKind = UncertaintyKind.LatentPosterior
    override fun predictJoint(features: DenseMatrix) = JointPredictiveMoments(
        DenseVector.of(DoubleArray(features.rows) { means[features[it, 0].toInt()] }),
        DenseMatrix.zero(features.rows).also { result ->
            for (i in 0 until features.rows) {
                for (j in 0 until features.rows) {
                    result[i, j] = covariance[features[i, 0].toInt()][features[j, 0].toInt()]
                }
            }
        },
    )
    override fun predict(features: DenseMatrix): PredictiveMoments {
        val joint = predictJoint(features)
        return PredictiveMoments(joint.means, DenseVector.of(DoubleArray(features.rows) { joint.covariance[it, it] }))
    }
    override fun drawFunction(random: Random): SampledFunction = error("not used")
}
