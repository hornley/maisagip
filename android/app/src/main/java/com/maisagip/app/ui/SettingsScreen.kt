package com.maisagip.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale

@Composable
fun SettingsScreen(confidence: Float, onConfidenceChanged: (Float) -> Unit) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Review threshold", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
            Text(String.format(Locale.ROOT, "%.2f", confidence), style = MaterialTheme.typography.titleMedium)
        }
        Slider(value = confidence, onValueChange = onConfidenceChanged, valueRange = 0.50f..0.80f)
    }
}
