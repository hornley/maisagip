package com.maisagip.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maisagip.app.data.EarResultEntity
import com.maisagip.app.inference.EarFinding
import java.io.File
import java.util.Locale

@Composable
fun EarResultScreen(result: EarResultEntity, onInspectMore: () -> Unit) {
    val finding = EarFinding.fromJson(result.detailJson)
    val preview = File(result.imageDirectory, finding.previewFileName)
    LazyColumn(
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(finding.gradeLabel, style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Bold)
                Text(varietyName(finding.varietyClass), style = MaterialTheme.typography.titleLarge)
                if (finding.needsReinspection) {
                    Surface(color = MaterialTheme.colorScheme.errorContainer, shape = MaterialTheme.shapes.small) {
                        Text("Review this result", modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp), color = MaterialTheme.colorScheme.onErrorContainer)
                    }
                }
            }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                PrivatePhoto(preview, "Annotated corn ear inspection", Modifier.fillMaxWidth().height(260.dp))
            }
        }
        item {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Metric("Defect coverage", percent(finding.defectCoverage))
                Metric("Kernel completeness", percent(finding.kernelCompleteness))
                Metric("Ear size", finding.earSize.replaceFirstChar { it.uppercase() })
            }
        }
        item { Text("Detected defects", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        if (finding.defects.isEmpty()) {
            item { Text("No supported defects detected", color = MaterialTheme.colorScheme.onSurfaceVariant) }
        } else {
            items(finding.defects, key = { it.className }) { defect ->
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("${defect.className.replace('_', ' ')} · ${defect.count}")
                    Text(percent(defect.coverage))
                }
            }
        }
        item { Text("Grade", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        items(finding.gradeReasons) { reason -> Text(reason, style = MaterialTheme.typography.bodyMedium) }
        item { Text("Utilization", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        item {
            Text("${finding.utilization.replace('_', ' ')} · ${finding.utilizationReason}", style = MaterialTheme.typography.bodyMedium)
        }
        if (finding.recommendations.isNotEmpty()) {
            item { Text("Recommendations", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
            items(finding.recommendations) { option -> Text(option.label, style = MaterialTheme.typography.bodyMedium) }
        }
        if (finding.reviewReasons.isNotEmpty()) {
            item { Text(finding.reviewReasons.joinToString(" "), color = MaterialTheme.colorScheme.error) }
        }
        item {
            Text("Visual findings do not determine food safety.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        item {
            Button(onClick = onInspectMore, modifier = Modifier.fillMaxWidth()) { Text("Inspect more") }
        }
    }
}

@Composable
private fun Metric(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
    }
}

private fun percent(value: Float) = String.format(Locale.ROOT, "%.1f%%", value * 100)
