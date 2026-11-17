package com.maisagip.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maisagip.app.data.HarvestEntity

@Composable
fun HarvestHistoryScreen(harvests: List<HarvestEntity>) {
    if (harvests.isEmpty()) {
        Column(modifier = Modifier.fillMaxWidth().padding(20.dp)) {
            Text("No completed harvests", style = MaterialTheme.typography.titleMedium)
        }
        return
    }
    LazyColumn(
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 16.dp),
    ) {
        items(harvests, key = { it.id }) { harvest ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(dateLabel(harvest.startedAtMillis), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text("${harvest.earCount} ears", style = MaterialTheme.typography.titleMedium)
                    }
                    Text(
                        "Extra ${harvest.extraCount} · Class I ${harvest.classICount} · Class II ${harvest.classIICount} · Reject ${harvest.rejectCount}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}
