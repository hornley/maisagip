package com.maisagip.app

import android.app.Application
import com.maisagip.app.data.HarvestDatabase
import com.maisagip.app.data.HarvestRepository
import com.maisagip.app.data.StagedMediaStore
import com.maisagip.app.inference.InspectionPipeline
import com.maisagip.app.inference.OnnxModels
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class MaisagipApplication : Application() {
    lateinit var database: HarvestDatabase
        private set
    lateinit var harvestRepository: HarvestRepository
        private set
    lateinit var stagedMediaStore: StagedMediaStore
        private set

    val onnxModels: OnnxModels by lazy(LazyThreadSafetyMode.SYNCHRONIZED) { OnnxModels.load(this) }
    val inspectionPipeline: InspectionPipeline by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        InspectionPipeline(this, onnxModels)
    }

    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        database = HarvestDatabase.get(this)
        harvestRepository = HarvestRepository(this, database)
        stagedMediaStore = StagedMediaStore(this)
        applicationScope.launch { harvestRepository.cleanupArchivedImages() }
    }
}
