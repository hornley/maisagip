package com.maisagip.app.inference

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.security.MessageDigest
import org.json.JSONObject

enum class ModelTask(val manifestKey: String) {
    SEGMENTER("segmenter"),
    CLASSIFIER("classifier"),
    DETECTOR("detector"),
}

data class TensorOutput(val shape: LongArray, val values: FloatArray)

private data class ModelSpec(
    val file: String,
    val sha256: String,
    val inputName: String,
    val outputNames: List<String>,
    val classNames: List<String>,
)

class OnnxModels private constructor(
    private val environment: OrtEnvironment,
    private val specs: Map<ModelTask, ModelSpec>,
    private val sessions: Map<ModelTask, OrtSession>,
) : AutoCloseable {
    fun classNames(task: ModelTask): List<String> = specs.getValue(task).classNames

    fun run(task: ModelTask, input: FloatArray, shape: LongArray): Map<String, TensorOutput> {
        val spec = specs.getValue(task)
        val session = sessions.getValue(task)
        val tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), shape)
        tensor.use {
            session.run(mapOf(spec.inputName to tensor)).use { result ->
                return spec.outputNames.associateWith { name ->
                    val output = result.get(name).orElseThrow {
                        IllegalStateException("${task.name.lowercase()} model output '$name' is missing.")
                    } as? OnnxTensor ?: error("${task.name.lowercase()} output '$name' is not a tensor.")
                    val info = output.info
                    val values = FloatArray(output.floatBuffer.remaining())
                    output.floatBuffer.duplicate().get(values)
                    TensorOutput(info.shape, values)
                }
            }
        }
    }

    override fun close() {
        sessions.values.forEach { it.close() }
    }

    companion object {
        fun load(context: Context): OnnxModels {
            val appContext = context.applicationContext
            val manifest = try {
                appContext.assets.open("models/model_manifest.json").bufferedReader().use {
                    JSONObject(it.readText())
                }
            } catch (error: Exception) {
                throw IllegalStateException("The bundled model manifest could not be read.", error)
            }
            val root = File(appContext.filesDir, "models")
            check(root.mkdirs() || root.isDirectory) { "Private model storage could not be created." }
            val environment = OrtEnvironment.getEnvironment()
            val specs = ModelTask.entries.associateWith { task ->
                val entry = manifest.getJSONObject("models").getJSONObject(task.manifestKey)
                ModelSpec(
                    file = entry.getString("file"),
                    sha256 = entry.getString("sha256"),
                    inputName = entry.getString("input_name"),
                    outputNames = entry.getJSONArray("output_names").let { names ->
                        (0 until names.length()).map(names::getString)
                    },
                    classNames = entry.getJSONArray("class_names").let { names ->
                        (0 until names.length()).map(names::getString)
                    },
                )
            }
            val sessions = linkedMapOf<ModelTask, OrtSession>()
            try {
                specs.forEach { (task, spec) ->
                    val assetName = "models/${spec.file}"
                    val modelFile = File(root, spec.file)
                    if (!modelFile.isFile || sha256(modelFile) != spec.sha256) {
                        val tempFile = File(root, "${spec.file}.partial")
                        appContext.assets.open(assetName).use { input ->
                            tempFile.outputStream().use { output -> input.copyTo(output) }
                        }
                        check(sha256(tempFile) == spec.sha256) { "The bundled ${task.name.lowercase()} model failed its checksum." }
                        modelFile.delete()
                        check(tempFile.renameTo(modelFile)) { "The bundled ${task.name.lowercase()} model could not be installed." }
                    }
                    val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
                    val session = try {
                        environment.createSession(modelFile.absolutePath, options)
                    } finally {
                        options.close()
                    }
                    check(spec.inputName in session.inputNames) {
                        "The bundled ${task.name.lowercase()} model has an unexpected input."
                    }
                    check(session.outputNames.containsAll(spec.outputNames)) {
                        "The bundled ${task.name.lowercase()} model has unexpected outputs."
                    }
                    sessions[task] = session
                }
            } catch (error: Exception) {
                sessions.values.forEach { it.close() }
                throw IllegalStateException("A required offline model could not be loaded: ${error.message}", error)
            }
            return OnnxModels(environment, specs, sessions)
        }

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().buffered().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
        }
    }
}
