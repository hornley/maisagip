package com.maisagip.app.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.maisagip.app.data.StagedMediaStore
import java.io.File

@Composable
fun MediaSelectionScreen(
    stagedStore: StagedMediaStore,
    files: List<File>,
    busy: Boolean,
    onFilesChanged: () -> Unit,
    onRemove: (File) -> Unit,
    onCamera: () -> Unit,
    onStartInspection: () -> Unit,
) {
    val context = LocalContext.current
    var error by remember { mutableStateOf<String?>(null) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia()) { uris ->
        if (uris.isNotEmpty()) {
            runCatching { stagedStore.replaceFromUris(uris) }
                .onSuccess { onFilesChanged() }
                .onFailure { error = it.message ?: "Could not read those photos." }
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(
                onClick = onCamera,
                enabled = !busy,
                modifier = Modifier.weight(1f).height(52.dp),
            ) { Text("Take a photo") }
            OutlinedButton(
                onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy,
                modifier = Modifier.weight(1f).height(52.dp),
            ) { Text("Choose photos") }
        }

        if (files.isNotEmpty()) {
            Card {
                Text(
                    if (files.size == 1) "One photo · visible ears will be separated automatically"
                    else "${files.size} photos · combined as views of one ear",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.fillMaxWidth().padding(14.dp),
                )
            }
            LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                items(files, key = { it.absolutePath }) { file ->
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Card(shape = RoundedCornerShape(12.dp)) {
                            PrivatePhoto(file, "Selected corn photo", Modifier.size(96.dp))
                        }
                        OutlinedButton(onClick = { onRemove(file) }, enabled = !busy) { Text("Remove") }
                    }
                }
            }
        }

        error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        if (files.isNotEmpty()) {
            Button(
                onClick = onStartInspection,
                enabled = !busy,
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text(if (busy) "Inspecting" else "Start inspection") }
        }
    }
}

@Composable
fun CameraCaptureScreen(
    stagedStore: StagedMediaStore,
    onCaptured: () -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var permissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var error by remember { mutableStateOf<String?>(null) }
    val permissionRequest = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permissionGranted = granted
        if (!granted) error = "Camera permission is needed to take a photo."
    }
    LaunchedEffect(Unit) {
        if (!permissionGranted) permissionRequest.launch(Manifest.permission.CAMERA)
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedButton(onClick = onBack) { Text("Back") }
        if (permissionGranted) {
            CameraPreview(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                onError = { error = it },
                onCaptureReady = { capture ->
                    error = null
                    val output = stagedStore.createCameraOutput()
                    val options = ImageCapture.OutputFileOptions.Builder(output).build()
                    capture.takePicture(
                        options,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                                stagedStore.addCameraOutput(output)
                                onCaptured()
                            }

                            override fun onError(exception: ImageCaptureException) {
                                output.delete()
                                error = exception.message ?: "The photo could not be saved."
                            }
                        },
                    )
                },
            )
        } else {
            Column(modifier = Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(error ?: "Allow camera access to take a photo.")
                Button(onClick = { permissionRequest.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
            }
        }
        error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
    }
}

@Composable
private fun CameraPreview(
    modifier: Modifier,
    onError: (String) -> Unit,
    onCaptureReady: (ImageCapture) -> Unit,
) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    var capture by remember { mutableStateOf<ImageCapture?>(null) }
    var provider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    val future = remember { ProcessCameraProvider.getInstance(context) }

    DisposableEffect(owner, future) {
        future.addListener({
            runCatching {
                val cameraProvider = future.get()
                provider = cameraProvider
                val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                val imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                capture = imageCapture
            }.onFailure { onError(it.message ?: "Camera could not be opened.") }
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            provider?.unbindAll()
            provider = null
        }
    }

    Box(modifier) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        Button(
            onClick = { capture?.let(onCaptureReady) },
            enabled = capture != null,
            modifier = Modifier.align(Alignment.BottomCenter).padding(20.dp),
        ) { Text("Capture") }
    }
}
