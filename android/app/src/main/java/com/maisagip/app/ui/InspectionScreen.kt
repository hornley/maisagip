package com.maisagip.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maisagip.app.InspectionUiState
import com.maisagip.app.inference.ViewPredictionSummary

@Composable
fun InspectionScreen(
    state: InspectionUiState,
    onChangePhotos: () -> Unit,
    onRetry: () -> Unit,
    onReturnHome: () -> Unit,
) {
    when (state) {
        InspectionUiState.Idle -> Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Ready to inspect", style = MaterialTheme.typography.titleLarge)
            Button(onClick = onRetry) { Text("Start inspection") }
        }
        InspectionUiState.Running -> Column(
            modifier = Modifier.fillMaxWidth().padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            CircularProgressIndicator()
            Text("Inspecting photos", style = MaterialTheme.typography.titleMedium)
        }
        is InspectionUiState.Failed -> Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text(state.message, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.error)
            Button(onClick = onChangePhotos) { Text("Change photos") }
            OutlinedButton(onClick = onReturnHome) { Text("Back to harvest") }
        }
        is InspectionUiState.NeedsReview -> VarietyReview(state.predictions, state.message, onChangePhotos, onReturnHome)
        is InspectionUiState.Saved -> Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Inspection saved", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Button(onClick = onReturnHome) { Text("View harvest") }
        }
    }
}

@Composable
private fun VarietyReview(
    predictions: List<ViewPredictionSummary>,
    message: String,
    onChangePhotos: () -> Unit,
    onReturnHome: () -> Unit,
) {
    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { Text(message, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        items(predictions) { result ->
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Photo ${result.number}: ${varietyName(result.variety)}")
                Text(String.format(java.util.Locale.ROOT, "%.0f%%", result.confidence * 100))
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onChangePhotos, modifier = Modifier.fillMaxWidth()) { Text("Review photos") }
                OutlinedButton(onClick = onReturnHome, modifier = Modifier.fillMaxWidth()) { Text("Back to harvest") }
            }
        }
    }
}
