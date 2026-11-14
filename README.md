# mediapipe-facemesh-action — Android build of MediaPipe's face+iris library

> **Credit.** Fork of [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh) (upstream commit `962bddab`), Apache-2.0 / BSD-3-Clause. Assembled by **Inkling 2 Small Free** (OpenCode agent, model `opencode/inkling-2-small-free`) under Windows81's direction. AI-authored under human direction — not pure human authorship. For the Flutter plugin, use cornpip's repo.

Three files, one goal: fast eye-region tracking in a plain Android app.

Five files, including three JNI builds of the same source.

- `libmediapipe_face_mesh.so` — the library that finds faces and computes landmarks (468 + 5 iris per eye).
- `libmpfm_tensorflowlite_c.so` — the TFLite runtime that actually runs the model.
- `libeygaze_mesh_jni.so` — default JNI bridge: the original **9-float** feature vector (ax ay bx by nx ny izm izd zNose). Drop-in for any existing caller.
- `libeygaze_mesh_jni15.so` — same source, extended **15-float** vector (the 9, plus eye aspect ratios, eye/inter-ocular widths, iris radii). Built specifically for the Surface Duo pipeline, where the tracking quality of the 9-float build was not adequate. The indices of the first 9 are identical between the two artifacts.
- `libeygaze_mesh_jni15_mir.so` — additive mirror variant of the 15-float bridge: same code path, same vector, same feature semantics, but the captured frame is mirrored horizontally before detection and landmark passes (`mirror_horizontal=1`). For rigs whose camera hands in a left-right swapped image to the model. Not the default; never silently activated.

## How do I use this?

1. Download the latest release zip and unzip it.
2. Put the three `.so` files into `app/src/main/jniLibs/arm64-v8a/` in your project.
3. Put `face_landmark_with_attention.tflite` and `face_detection_full_range.tflite` into `app/src/main/assets/`.
4. Load the libraries once anywhere:
   ```kotlin
   System.loadLibrary("mpfm_tensorflowlite_c")
   System.loadLibrary("mediapipe_face_mesh")
   System.loadLibrary("eygaze_mesh_jni")
   ```
5. Use it (picks one):
   - **JNI bridge** (fast, opinionated): write a tiny class `EyegazeFaceMeshNative` matching `src/eygaze_mesh_jni.cc`, call `nativeCreate`, `nativeFeatureExtractNv21`, `nativeDestroy`.
   - **Plain C** (raw): use `mp_face_mesh_create`, `mp_face_mesh_process_nv21`, `mp_face_mesh_release_result`, `mp_face_mesh_destroy` directly from your own JNI shim.

## Where can I put this?

- **A reading app** — knows which word the eyes land on.
- **An accessibility / gaze cursor** — eye drives a pointer.
- **An assisted-input example** — look left to dodge, look at a card to select.
- **A fatigue detector** — alert after long eyelid droop.

## What functions can I call?

**JNI bridge** (from Kotlin/Java):

- `nativeCreate(modelPath: String, detectorPath: String): Long` — start tracking. Returns a handle; 0 when either model fails to initialise.
- `nativeFeatureExtractNv21(handle, yBuf, vuBuf, width, height, yRowStride, vuRowStride, rotationDegrees): FloatArray` — one NV21 camera frame in, 15-float vector out for `libeygaze_mesh_jni15.so` (9-float for `libeygaze_mesh_jni.so`); NaN-filled array when no face.
- `nativeDestroy(handle: Long)` — stop tracking.

**Plain C** (via your own JNI shim):

- `mp_face_mesh_create` — start a mesh context.
- `mp_face_mesh_process_nv21` — feed NV21, get landmarks.
- `mp_face_mesh_release_result` — free the result.
- `mp_face_detector_create` — start a BlazeFace full-range context (required: the attention model needs its rect).
- `mp_face_detector_process_nv21` — feed NV21, get face detections (use `expanded_face_rect` as the mesh override).
- `mp_face_detector_release_result` — free the detections.
- `mp_face_mesh_last_error` — what went wrong.
- `mp_face_mesh_destroy` — destroy the context.

## Working example (real, in-app)

This is the class `com.eyegaze` uses, assembled by Inkling 2 Small Free under Windows81's direction, trimmed only on import:

```kotlin
final class EyegazeFaceMeshNative(Context ctx) : AutoCloseable {
    init {
        System.loadLibrary("mpfm_tensorflowlite_c")
        System.loadLibrary("mediapipe_face_mesh")
        System.loadLibrary("eygaze_mesh_jni")
    }
    private var handle: Long = 0L

    constructor(ctx: Context) : this(ctx) {
        copyModelFromAssetsToFiles(ctx)          // copies BOTH .tflite files
        handle = nativeCreate(
            File(ctx.filesDir, MODEL_FILE).absolutePath,
            File(ctx.filesDir, DETECTOR_FILE).absolutePath)
        check(handle != 0L) { "nativeCreate failed — see 'nativeCreate' log (tag:eyegaze)" }
    }

    fun extract(yBuffer: ByteBuffer, vuBuffer: ByteBuffer,
                width: Int, height: Int,
                yRowStride: Int, vuRowStride: Int, rotationDegrees: Int): FloatArray? {
        if (handle == 0L) return null
        val features = nativeFeatureExtractNv21(handle, yBuffer, vuBuffer,
                width, height, yRowStride, vuRowStride, rotationDegrees)
        if (features == null) return null
        return if (features.all { it.isNaN() }) null else features
    }

    override fun close() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    private external fun nativeCreate(modelPath: String, detectorPath: String): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeFeatureExtractNv21(
        handle: Long, yBuffer: ByteBuffer, vuBuffer: ByteBuffer,
        width: Int, height: Int, yRowStride: Int, vuRowStride: Int,
        rotationDegrees: Int): FloatArray
}
```

And this is the camera callback used by `EyeGazeService`:

```kotlin
private val bridge = EyegazeFaceMeshNative(this)

private fun onFrame(image: Image, now: Long) {
    image.planes.let { planes ->
        val y = planes[0].buffer
        val vu = planes[2].buffer   // NV21 layout: Y plane followed by interleaved VU
        val features = bridge.extract(y, vu, image.width, image.height,
            planes[0].rowStride, planes[2].rowStride, 270)   // 270° sensor orientation
        if (features != null) {
            LAST_FEATURE = features
        }
    }
}
```

## Why two models, every frame

`face_landmark_with_attention.tflite` is the refinement stage of MediaPipe's face pipeline: it expects a crop around an already-detected face, not the full camera frame. Feed it the whole frame and the face flag inside the model never clears the presence threshold — every frame reads all-NaN. The bridge therefore runs a BlazeFace full-range pass (`face_detection_full_range.tflite`) on the full frame first, takes the highest-scoring detection, and feeds `expanded_face_rect` from that detection to the landmark model. Both models ship as assets; both are copied to `filesDir` on first launch before `nativeCreate`.

Consequences worth knowing:

- `nativeCreate` returns 0 if *either* model fails to initialise — the attention model without its detector would silently never find a face, so the bridge refuses to run half-wired.
- The detector pass adds a few milliseconds per frame. If you don't need iris/eye features every frame, run detection on a timer and reuse the last rect while it is fresh.
- The scene is inferred, never captured: nothing in this library writes frames to disk or to the network.

## Reading the native logs

The bridge logs through Android's tag `eyegaze` (Java) and `MP_LOGI`/`MP_LOGE` (C) at boot:

```
I eyegaze : native libraries loaded              <- all three .so files resolved
I EyeGaze : EyegazeFaceMeshNative ready           <- the JNI bridge constructed
D eyegaze : nativeFeatureExtractNv21: handle=... <- one line per frame
W eyegaze : nativeFeatureExtractNv21 all-NaN     <- no face found this frame
W EyeGaze : EyegazeFaceMeshNative init failed     <- bad model file or load error
```

Watch for the two distinct failure modes:

- **all-NaN every frame** — usually a bad rotation (the model wants the face oriented
  upright for the sensor you used), a missing model file path (the bridge copies
  `face_landmark_with_attention.tflite` from `assets/` into `filesDir`
  on first use), or insufficient contrast (very low light reads all-NaN).
- **init failed** — the `.so` chain didn't load. If you see this, check
  `libmpfm_tensorflowlite_c.so` is under `jniLibs/<your-abi>/` next to `libmediapipe_face_mesh.so`.

## Feature vector — two lengths, same order

`nativeFeatureExtractNv21` fills and returns:

| index | meaning |
|-------|---------|
| 0, 1 | `ax, ay` — iris offset A relative to eye A (normalized units) |
| 2, 3 | `bx, by` — iris offset B relative to eye B |
| 4, 5 | `nx, ny` — nose-tip offset relative to the between-eyes midpoint, / inter-ocular distance |
| 6, 7 | `izm, izd` — mean / difference of iris z relative to eye-contour depth |
| 8 | `zNose` — nose-tip depth relative to eye depth |
| 9, 10 | `earA, earB` — eye aspect ratios (h/w), used for blink gating and as openness signal |
| 11, 12 | `wAN, wBN` — eye A / B width divided by inter-ocular distance |
| 13, 14 | `irisRA, irisRB` — iris radii (mean distance of the 5 iris points from their centroid) / eye width |

`libeygaze_mesh_jni.so` (the default bridge) returns indices 0-8 only — the original schema. A caller built for 9 floats reads those same 0-8 from the 15-float array and behaves identically; the indices never change meaning between the two artifacts.

## Minimal example

```kotlin
val handle = EyegazeFaceMeshNative.nativeCreate(
    filesDir.resolve("face_landmark_with_attention.tflite").absolutePath,
    filesDir.resolve("face_detection_full_range.tflite").absolutePath)
val features = EyegazeFaceMeshNative.nativeFeatureExtractNv21(
    handle, yBuf, vuBuf, width, height, yRowStride, vuRowStride, rotationDegrees)
EyegazeFaceMeshNative.nativeDestroy(handle)
```

`features` is 9 floats: the same feature vector `com.eyegaze` feeds to its gaze regressor, assembled by agentic engineering under Windows81's direction. Read them, feed them to your own regressor, done.

## Honest limits

- 640×480 or 64×480 camera frames only (front-camera software pipeline ceiling).
- 18–25 fps realistic.
- Eye-distance works at ~25–45 cm.
- Not production-hardened. Treat as a working build, not a hardened product.

## If you want to build the library yourself (not normal)

Release zips already ship the artifacts. The working recipe is `.github/workflows/main.yml` (NDK r26d, CMake, arm64-v8a + x86_64).

## License and attribution

Apache-2.0 / BSD-3-Clause. Not an authoritative upstream — for the Flutter plugin, see [cornpip's repository](https://github.com/cornpip/mediapipe_face_mesh).
