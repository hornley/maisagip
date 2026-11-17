package com.maisagip.app.data

import android.content.Context
import android.net.Uri
import java.io.File
import java.util.UUID

class StagedMediaStore(context: Context) {
    private val appContext = context.applicationContext
    private val stagingRoot = File(appContext.filesDir, "staging")
    private val preferences = appContext.getSharedPreferences("staged_media", Context.MODE_PRIVATE)

    fun currentFiles(): List<File> {
        val id = preferences.getString(KEY_ID, null) ?: return emptyList()
        return File(stagingRoot, id).listFiles()
            ?.filter(File::isFile)
            ?.sortedBy(File::getName)
            .orEmpty()
    }

    fun replaceFromUris(uris: List<Uri>): List<File> {
        require(uris.isNotEmpty()) { "Choose at least one photo." }
        clear()
        val id = UUID.randomUUID().toString()
        val directory = File(stagingRoot, id)
        check(directory.mkdirs()) { "Could not prepare private photo storage." }
        try {
            uris.forEachIndexed { index, uri ->
                val extension = when (appContext.contentResolver.getType(uri)?.lowercase()) {
                    "image/png" -> "png"
                    "image/heic" -> "heic"
                    "image/heif" -> "heif"
                    else -> "jpg"
                }
                val destination = File(directory, "%03d.%s".format(index + 1, extension))
                val input = appContext.contentResolver.openInputStream(uri)
                    ?: error("Could not read the selected photo.")
                input.use { source -> destination.outputStream().use { output -> source.copyTo(output) } }
            }
            check(preferences.edit().putString(KEY_ID, id).commit()) {
                "Could not save the selected photos."
            }
            return currentFiles()
        } catch (error: Exception) {
            directory.deleteRecursively()
            throw error
        }
    }

    fun createCameraOutput(): File {
        val directory = currentDirectory(createIfMissing = true)
        return File(directory, "%03d.jpg".format((directory.listFiles()?.size ?: 0) + 1))
    }

    fun addCameraOutput(file: File) {
        val directory = currentDirectory(createIfMissing = true)
        if (file.parentFile != directory) {
            val target = File(directory, "%03d.jpg".format((directory.listFiles()?.size ?: 0) + 1))
            file.copyTo(target, overwrite = true)
            file.delete()
        }
        preferences.edit().putString(KEY_ID, directory.name).commit()
    }

    fun remove(file: File) {
        file.delete()
        if (currentFiles().isEmpty()) clear()
    }

    fun moveTo(directory: File): List<File> {
        val sources = currentFiles()
        check(sources.isNotEmpty()) { "Select at least one photo." }
        check(directory.mkdirs() || directory.isDirectory) { "Could not prepare harvest photo storage." }
        return sources.map { source ->
            val target = File(directory, source.name)
            source.copyTo(target, overwrite = false)
            target
        }
    }

    fun clear() {
        val id = preferences.getString(KEY_ID, null)
        if (id != null) File(stagingRoot, id).deleteRecursively()
        preferences.edit().clear().commit()
    }

    private fun currentDirectory(createIfMissing: Boolean): File {
        val existingId = preferences.getString(KEY_ID, null)
        val id = existingId ?: UUID.randomUUID().toString()
        val directory = File(stagingRoot, id)
        if (createIfMissing) check(directory.mkdirs() || directory.isDirectory) {
            "Could not prepare private photo storage."
        }
        if (existingId == null && createIfMissing) {
            check(preferences.edit().putString(KEY_ID, id).commit()) {
                "Could not save the photo selection."
            }
        }
        return directory
    }

    companion object {
        private const val KEY_ID = "current_id"
    }
}
