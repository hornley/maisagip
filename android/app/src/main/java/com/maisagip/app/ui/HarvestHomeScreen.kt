package com.maisagip.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maisagip.app.data.EarResultEntity
import com.maisagip.app.data.HarvestEntity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HarvestHomeScreen(
    harvest: HarvestEntity?,
    results: List<EarResultEntity>,
    onStartHarvest: () -> Unit,
    onInspect: () -> Unit,
    onStartNextHarvest: () -> Unit,
    onOpenResult: (EarResultEntity) -> Unit,
) {
    LazyColumn(
        verticalArrangement = Arrangement.spacedBy(16.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        if (harvest == null) {
            item {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Ready to inspect", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                    Button(onClick = onStartHarvest, modifier = Modifier.fillMaxWidth()) {
                        Text("Start harvest")
                    }
                }
            }
        } else {
            item {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                        Column {
                            Text("Current harvest", style = MaterialTheme.typography.titleMedium)
                            Text(dateLabel(harvest.startedAtMillis), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        Text("${harvest.earCount} ears", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    }
                    Button(onClick = onInspect, modifier = Modifier.fillMaxWidth()) {
                        Text("Inspect more corn")
                    }
                    OutlinedButton(onClick = onStartNextHarvest, modifier = Modifier.fillMaxWidth()) {
                        Text("Start a new harvest")
                    }
                }
            }
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    GradeCount("Extra", harvest.extraCount, Modifier.weight(1f))
                    GradeCount("Class I", harvest.classICount, Modifier.weight(1f))
                    GradeCount("Class II", harvest.classIICount, Modifier.weight(1f))
                    GradeCount("Reject", harvest.rejectCount, Modifier.weight(1f))
                }
            }
            if (results.isEmpty()) {
                item { Text("No ears yet", color = MaterialTheme.colorScheme.onSurfaceVariant) }
            } else {
                item { Text("Inspected ears", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium) }
                items(results, key = { it.id }) { result -> EarResultRow(result, onOpen = { onOpenResult(result) }) }
            }
        }
    }
}

@Composable
private fun GradeCount(label: String, count: Int, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface, shape = MaterialTheme.shapes.medium) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(vertical = 12.dp, horizontal = 4.dp)) {
            Text(count.toString(), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
fun EarResultRow(result: EarResultEntity, onOpen: () -> Unit) {
    val finding = runCatching { com.maisagip.app.inference.EarFinding.fromJson(result.detailJson) }.getOrNull()
    Card(modifier = Modifier.fillMaxWidth().clickable(onClick = onOpen)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(finding?.gradeLabel ?: result.grade, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(varietyName(result.variety), style = MaterialTheme.typography.bodyMedium)
                if (result.needsReinspection) Text("Review needed", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.error)
            }
            Text("›", style = MaterialTheme.typography.headlineSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

fun dateLabel(epochMillis: Long): String = SimpleDateFormat("dd MMM yyyy", Locale.getDefault()).format(Date(epochMillis))

fun varietyName(value: String): String = when (value) {
    "yellow_sweet_corn" -> "Sweet Fortune"
    "white_corn" -> "Sweet Pearl"
    else -> value.replace('_', ' ').replaceFirstChar { it.uppercase() }
}
