package com.maisagip.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface HarvestDao {
    @Query("SELECT * FROM harvests WHERE archivedAtMillis IS NULL ORDER BY startedAtMillis DESC LIMIT 1")
    fun observeActiveHarvest(): Flow<HarvestEntity?>

    @Query("SELECT * FROM harvests WHERE archivedAtMillis IS NULL ORDER BY startedAtMillis DESC LIMIT 1")
    suspend fun getActiveHarvest(): HarvestEntity?

    @Query("SELECT * FROM harvests WHERE archivedAtMillis IS NOT NULL ORDER BY startedAtMillis DESC")
    fun observeArchivedHarvests(): Flow<List<HarvestEntity>>

    @Query("SELECT * FROM harvests WHERE archivedAtMillis IS NOT NULL ORDER BY startedAtMillis DESC")
    suspend fun getArchivedHarvests(): List<HarvestEntity>

    @Query("SELECT * FROM ear_results WHERE harvestId = :harvestId ORDER BY inspectedAtMillis DESC")
    fun observeEarResults(harvestId: String): Flow<List<EarResultEntity>>

    @Query("SELECT * FROM ear_results WHERE harvestId = :harvestId ORDER BY inspectedAtMillis DESC")
    suspend fun getEarResults(harvestId: String): List<EarResultEntity>

    @Query("SELECT imageDirectory FROM ear_results")
    suspend fun getReferencedImageDirectories(): List<String>

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertHarvest(harvest: HarvestEntity)

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertEarResults(results: List<EarResultEntity>)

    @Update
    suspend fun updateHarvest(harvest: HarvestEntity)

    @Query("DELETE FROM ear_results WHERE harvestId = :harvestId")
    suspend fun deleteEarResults(harvestId: String)

    suspend fun archiveHarvest(harvest: HarvestEntity, nextHarvest: HarvestEntity) {
        updateHarvest(harvest)
        deleteEarResults(harvest.id)
        insertHarvest(nextHarvest)
    }
}
