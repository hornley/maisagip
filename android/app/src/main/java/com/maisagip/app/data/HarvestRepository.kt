package com.maisagip.app.data

import android.content.Context
import androidx.room.withTransaction
import com.maisagip.app.inference.EarFinding
import java.io.File
import java.util.UUID
import kotlinx.coroutines.flow.Flow

class HarvestRepository(context: Context, private val database: HarvestDatabase) {
    private val appContext = context.applicationContext
    private val dao = database.harvestDao()
    private val harvestRoot = File(appContext.filesDir, "harvests")
    val activeHarvest: Flow<HarvestEntity?> = dao.observeActiveHarvest()
    val archivedHarvests: Flow<List<HarvestEntity>> = dao.observeArchivedHarvests()

    fun results(harvestId: String): Flow<List<EarResultEntity>> = dao.observeEarResults(harvestId)

    suspend fun ensureActiveHarvest(): HarvestEntity {
        dao.getActiveHarvest()?.let { return it }
        val harvest = HarvestEntity(UUID.randomUUID().toString(), System.currentTimeMillis())
        dao.insertHarvest(harvest)
        return harvest
    }

    suspend fun saveInspection(findings: List<EarFinding>, stagedFiles: List<File>): List<EarResultEntity> {
        require(findings.isNotEmpty()) { "Inspection did not find a usable ear." }
        require(stagedFiles.isNotEmpty()) { "Select at least one photo." }
        val harvest = dao.getActiveHarvest() ?: ensureActiveHarvest()
        val inspectionId = UUID.randomUUID().toString()
        val directory = File(harvestRoot, "${harvest.id}/$inspectionId")
        check(directory.mkdirs() || directory.isDirectory) { "Could not prepare harvest photo storage." }

        try {
            stagedFiles.forEach { source -> source.copyTo(File(directory, source.name), overwrite = false) }
            findings.forEach { finding ->
                val preview = finding.previewJpeg ?: error("Inspection preview is missing.")
                File(directory, finding.previewFileName).writeBytes(preview)
            }
            val now = System.currentTimeMillis()
            val rows = findings.map { finding ->
                EarResultEntity(
                    id = UUID.randomUUID().toString(),
                    harvestId = harvest.id,
                    inspectedAtMillis = now,
                    grade = finding.gradeKey,
                    variety = finding.varietyClass,
                    needsReinspection = finding.needsReinspection,
                    detailJson = finding.toJson(),
                    imageDirectory = directory.absolutePath,
                )
            }
            database.withTransaction {
                val current = dao.getActiveHarvest()
                check(current?.id == harvest.id) { "The active harvest changed during inspection." }
                dao.insertEarResults(rows)
                val allResults = dao.getEarResults(harvest.id)
                dao.updateHarvest(harvest.copy(
                    earCount = allResults.size,
                    extraCount = allResults.count { it.grade == "ExtraClass" },
                    classICount = allResults.count { it.grade == "ClassI" },
                    classIICount = allResults.count { it.grade == "ClassII" },
                    rejectCount = allResults.count { it.grade == "Reject" },
                ))
            }
            return rows
        } catch (error: Exception) {
            directory.deleteRecursively()
            throw error
        }
    }

    suspend fun rolloverAndStartNext(): HarvestEntity {
        val old = dao.getActiveHarvest()
        if (old == null) return ensureActiveHarvest()

        val next = HarvestEntity(UUID.randomUUID().toString(), System.currentTimeMillis())
        database.withTransaction {
            val current = dao.getActiveHarvest()
            check(current?.id == old.id) { "The active harvest changed before rollover." }
            val results = dao.getEarResults(old.id)
            val archived = current.copy(
                archivedAtMillis = System.currentTimeMillis(),
                earCount = results.size,
                extraCount = results.count { it.grade == "ExtraClass" },
                classICount = results.count { it.grade == "ClassI" },
                classIICount = results.count { it.grade == "ClassII" },
                rejectCount = results.count { it.grade == "Reject" },
            )
            dao.archiveHarvest(archived, next)
        }

        File(harvestRoot, old.id).deleteRecursively()
        cleanupArchivedImages()
        return next
    }

    suspend fun cleanupArchivedImages() {
        val referenced = dao.getReferencedImageDirectories().map { File(it) }.toSet()
        val archivedIds = dao.getArchivedHarvests().map { it.id }.toSet()
        harvestRoot.listFiles()?.forEach { harvestDirectory ->
            if (harvestDirectory.name in archivedIds) harvestDirectory.deleteRecursively()
            else harvestDirectory.listFiles()?.forEach { inspectionDirectory ->
                if (inspectionDirectory !in referenced) inspectionDirectory.deleteRecursively()
            }
        }
    }

}
