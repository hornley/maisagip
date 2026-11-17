package com.maisagip.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.maisagip.app.data.EarResultEntity
import com.maisagip.app.data.HarvestEntity
import com.maisagip.app.inference.InconsistentVarietyException
import com.maisagip.app.inference.ViewPredictionSummary
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface InspectionUiState {
    data object Idle : InspectionUiState
    data object Running : InspectionUiState
    data class Saved(val results: List<EarResultEntity>) : InspectionUiState
    data class NeedsReview(val predictions: List<ViewPredictionSummary>, val message: String) : InspectionUiState
    data class Failed(val message: String) : InspectionUiState
}

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as MaisagipApplication
    private val preferences = application.getSharedPreferences("inspection_settings", 0)

    val activeHarvest: StateFlow<HarvestEntity?> = app.harvestRepository.activeHarvest.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        null,
    )
    val archivedHarvests: StateFlow<List<HarvestEntity>> = app.harvestRepository.archivedHarvests.stateIn(
        viewModelScope,
        SharingStarted.WhileSubscribed(5_000),
        emptyList(),
    )

    @OptIn(ExperimentalCoroutinesApi::class)
    val activeResults: StateFlow<List<EarResultEntity>> = activeHarvest.flatMapLatest { harvest ->
        if (harvest == null) flowOf(emptyList()) else app.harvestRepository.results(harvest.id)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private val _stagedFiles = MutableStateFlow(app.stagedMediaStore.currentFiles())
    val stagedFiles: StateFlow<List<File>> = _stagedFiles.asStateFlow()

    private val _inspectionState = MutableStateFlow<InspectionUiState>(InspectionUiState.Idle)
    val inspectionState: StateFlow<InspectionUiState> = _inspectionState.asStateFlow()

    private val _operationError = MutableStateFlow<String?>(null)
    val operationError: StateFlow<String?> = _operationError.asStateFlow()

    private val _confidenceThreshold = MutableStateFlow(
        preferences.getFloat(KEY_CONFIDENCE, 0.50f).coerceIn(0.50f, 0.80f),
    )
    val confidenceThreshold: StateFlow<Float> = _confidenceThreshold.asStateFlow()

    fun setConfidenceThreshold(value: Float) {
        val safeValue = value.coerceIn(0.50f, 0.80f)
        _confidenceThreshold.value = safeValue
        preferences.edit().putFloat(KEY_CONFIDENCE, safeValue).apply()
    }

    fun startHarvest() = viewModelScope.launch {
        _operationError.value = null
        runCatching { withContext(Dispatchers.IO) { app.harvestRepository.ensureActiveHarvest() } }
            .onFailure { _operationError.value = it.message ?: "Could not start a harvest." }
    }

    fun rolloverAndStartNext() = viewModelScope.launch {
        _operationError.value = null
        runCatching {
            withContext(Dispatchers.IO) { app.harvestRepository.rolloverAndStartNext() }
            app.stagedMediaStore.clear()
            refreshStagedFiles()
        }.onFailure { _operationError.value = it.message ?: "Could not save the harvest summary." }
    }

    fun inspect() = viewModelScope.launch {
        val files = app.stagedMediaStore.currentFiles()
        if (files.isEmpty()) {
            _inspectionState.value = InspectionUiState.Failed("Choose at least one photo.")
            return@launch
        }
        _inspectionState.value = InspectionUiState.Running
        try {
            val findings = withContext(Dispatchers.Default) {
                app.inspectionPipeline.inspect(files, _confidenceThreshold.value)
            }
            val rows = withContext(Dispatchers.IO) {
                app.harvestRepository.saveInspection(findings, files)
            }
            app.stagedMediaStore.clear()
            refreshStagedFiles()
            _inspectionState.value = InspectionUiState.Saved(rows)
        } catch (error: InconsistentVarietyException) {
            _inspectionState.value = InspectionUiState.NeedsReview(error.predictions, error.message.orEmpty())
        } catch (error: Exception) {
            _inspectionState.value = InspectionUiState.Failed(error.message ?: "Inspection could not be completed.")
        }
    }

    fun refreshStagedFiles() {
        _stagedFiles.value = app.stagedMediaStore.currentFiles()
    }

    fun replaceStagedFiles(files: List<File>) {
        _stagedFiles.value = files
    }

    fun clearInspectionState() {
        _inspectionState.value = InspectionUiState.Idle
    }

    fun dismissOperationError() {
        _operationError.value = null
    }

    companion object {
        private const val KEY_CONFIDENCE = "confidence_threshold"
    }
}
