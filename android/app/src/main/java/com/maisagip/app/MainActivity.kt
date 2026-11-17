package com.maisagip.app

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.maisagip.app.ui.CameraCaptureScreen
import com.maisagip.app.ui.EarResultScreen
import com.maisagip.app.ui.HarvestHistoryScreen
import com.maisagip.app.ui.HarvestHomeScreen
import com.maisagip.app.ui.InspectionScreen
import com.maisagip.app.ui.MediaSelectionScreen
import com.maisagip.app.ui.SettingsScreen
import com.maisagip.app.ui.theme.MaisagipTheme

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaisagipTheme {
                MaisagipAppContent()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MaisagipAppContent(viewModel: MainViewModel = viewModel()) {
    val app = androidx.compose.ui.platform.LocalContext.current.applicationContext as MaisagipApplication
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val route = backStackEntry?.destination?.route.orEmpty()
    val harvest by viewModel.activeHarvest.collectAsState()
    val ears by viewModel.activeResults.collectAsState()
    val history by viewModel.archivedHarvests.collectAsState()
    val stagedFiles by viewModel.stagedFiles.collectAsState()
    val inspectionState by viewModel.inspectionState.collectAsState()
    val operationError by viewModel.operationError.collectAsState()
    val confidence by viewModel.confidenceThreshold.collectAsState()
    var confirmRollover by remember { mutableStateOf(false) }

    LaunchedEffect(inspectionState) {
        val saved = inspectionState as? InspectionUiState.Saved
        val first = saved?.results?.firstOrNull()
        if (first != null) {
            navController.navigate("ear/${first.id}") {
                popUpTo("inspection") { inclusive = true }
                launchSingleTop = true
            }
        }
    }

    val tabs = listOf("harvest" to "Harvest", "history" to "History", "settings" to "Settings")
    val isTab = tabs.any { it.first == route }
    val title = when (route) {
        "harvest" -> "Maisagip"
        "history" -> "History"
        "settings" -> "Settings"
        "media" -> "Add photos"
        "camera" -> "Camera"
        "inspection" -> if (inspectionState is InspectionUiState.NeedsReview) "Review photos" else "Inspection"
        "ear/{id}" -> "Ear result"
        else -> "Maisagip"
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(title) },
                navigationIcon = {
                    if (!isTab) TextButton(onClick = { navController.popBackStack() }) { Text("Back") }
                },
            )
        },
        bottomBar = {
            if (isTab) NavigationBar {
                tabs.forEach { (tabRoute, label) ->
                    NavigationBarItem(
                        selected = backStackEntry?.destination?.hierarchy?.any { it.route == tabRoute } == true,
                        onClick = {
                            navController.navigate(tabRoute) {
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = {},
                        label = { Text(label) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(navController = navController, startDestination = "harvest", modifier = Modifier.padding(padding)) {
            composable("harvest") {
                HarvestHomeScreen(
                    harvest = harvest,
                    results = ears,
                    onStartHarvest = viewModel::startHarvest,
                    onInspect = {
                        viewModel.clearInspectionState()
                        navController.navigate("media")
                    },
                    onStartNextHarvest = {
                        if (harvest == null) viewModel.startHarvest() else confirmRollover = true
                    },
                    onOpenResult = { result -> navController.navigate("ear/${result.id}") },
                )
            }
            composable("history") { HarvestHistoryScreen(history) }
            composable("settings") { SettingsScreen(confidence, viewModel::setConfidenceThreshold) }
            composable("media") {
                MediaSelectionScreen(
                    stagedStore = app.stagedMediaStore,
                    files = stagedFiles,
                    busy = inspectionState == InspectionUiState.Running,
                    onFilesChanged = viewModel::refreshStagedFiles,
                    onRemove = { file -> app.stagedMediaStore.remove(file); viewModel.refreshStagedFiles() },
                    onCamera = { navController.navigate("camera") },
                    onStartInspection = {
                        viewModel.inspect()
                        navController.navigate("inspection")
                    },
                )
            }
            composable("camera") {
                CameraCaptureScreen(
                    stagedStore = app.stagedMediaStore,
                    onCaptured = { viewModel.refreshStagedFiles(); navController.popBackStack() },
                    onBack = { navController.popBackStack() },
                )
            }
            composable("inspection") {
                InspectionScreen(
                    state = inspectionState,
                    onChangePhotos = {
                        viewModel.clearInspectionState()
                        navController.navigate("media")
                    },
                    onRetry = viewModel::inspect,
                    onReturnHome = {
                        viewModel.clearInspectionState()
                        navController.navigate("harvest") {
                            popUpTo("harvest") { inclusive = false }
                            launchSingleTop = true
                        }
                    },
                )
            }
            composable("ear/{id}") { entry ->
                val id = entry.arguments?.getString("id")
                val result = ears.firstOrNull { it.id == id }
                if (result == null) {
                    Text("This result is no longer available.", modifier = Modifier.padding(20.dp))
                } else {
                    EarResultScreen(result, onInspectMore = {
                        navController.navigate("media") { launchSingleTop = true }
                    })
                }
            }
        }
    }

    if (confirmRollover && harvest != null) {
        AlertDialog(
            onDismissRequest = { confirmRollover = false },
            title = { Text("Start a new harvest?") },
            text = { Text("Photos and individual results from this harvest will be removed. Its summary will stay on this device.") },
            confirmButton = {
                TextButton(onClick = { confirmRollover = false; viewModel.rolloverAndStartNext() }) {
                    Text("Start new harvest", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = { TextButton(onClick = { confirmRollover = false }) { Text("Cancel") } },
        )
    }
    operationError?.let { message ->
        AlertDialog(
            onDismissRequest = viewModel::dismissOperationError,
            title = { Text("Could not save harvest") },
            text = { Text(message) },
            confirmButton = { TextButton(onClick = viewModel::dismissOperationError) { Text("OK") } },
        )
    }
}
