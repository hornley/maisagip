package com.maisagip.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.Room
import androidx.room.RoomDatabase

@Entity(tableName = "harvests")
data class HarvestEntity(
    @PrimaryKey val id: String,
    val startedAtMillis: Long,
    val archivedAtMillis: Long? = null,
    val earCount: Int = 0,
    val extraCount: Int = 0,
    val classICount: Int = 0,
    val classIICount: Int = 0,
    val rejectCount: Int = 0,
)

@Entity(tableName = "ear_results", primaryKeys = ["id"])
data class EarResultEntity(
    val id: String,
    val harvestId: String,
    val inspectedAtMillis: Long,
    val grade: String,
    val variety: String,
    val needsReinspection: Boolean,
    val detailJson: String,
    val imageDirectory: String,
)

@Database(entities = [HarvestEntity::class, EarResultEntity::class], version = 1)
abstract class HarvestDatabase : RoomDatabase() {
    abstract fun harvestDao(): HarvestDao

    companion object {
        @Volatile
        private var instance: HarvestDatabase? = null

        fun get(context: Context): HarvestDatabase = instance ?: synchronized(this) {
            instance ?: Room.databaseBuilder(
                context.applicationContext,
                HarvestDatabase::class.java,
                "maisagip.db",
            ).build().also { instance = it }
        }
    }
}
