package com.maisagip.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = GreenPrimary,
    onPrimary = GreenOnPrimary,
    secondary = HarvestGold,
    onSecondary = GreenOnPrimary,
    background = GreenBackground,
    onBackground = GreenText,
    surface = GreenSurface,
    onSurface = GreenText,
    error = ErrorRed,
)

private val DarkColors = darkColorScheme(
    primary = GreenPrimaryDark,
    onPrimary = GreenBackgroundDark,
    secondary = Color(0xFFFBBF24),
    onSecondary = GreenBackgroundDark,
    background = GreenBackgroundDark,
    onBackground = GreenTextDark,
    surface = GreenSurfaceDark,
    onSurface = GreenTextDark,
    error = Color(0xFFFFB4AB),
)

@Composable
fun MaisagipTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content,
    )
}
