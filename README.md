# mediapipe-facemesh-action — Android build of MediaPipe's face+iris library

> **Credit.** Forked arrangement of [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh) (upstream commit `962bddab`), Apache-2.0 / BSD-3-Clause. Assembly by **Inkling 2 Small Free** (OpenCode agent, model `opencode/inkling-2-small-free`). This is **agentic engineering** — AI-authored under human direction, not pure human authorship. Not an authoritative upstream — for the Flutter plugin, use cornpip's repository.
>
> If you know your way around Android Studio — Run button, drag-and-drop, reading a crash log — you can use this. No computer-vision or native-build background needed.

---

## What it is

Three files, taken as a whole, give a plain Android app fast eye-region tracking:

- `libmediapipe_face_mesh.so` — the compiled library that finds faces and computes landmarks. A `.so` file is a native library Android loads at runtime; Android Studio packs everything into the final app, and you never look inside it.
- `libmpfm_tensorflowlite_c.so` — the TensorFlow Lite runtime (the engine that executes the neural network), prebuilt bit for bit from upstream.
- `libeygaze_mesh_jni.so` — an optional bridge that lets Kotlin or Java call the library directly.

Given one camera frame, the library answers:

1. **Where is the face?** Which pixels belong to it, and where the eyes, nose and mouth are.
2. **Where are the pupils?** 478 named points per face — the standard 468-point face mesh plus 10 iris points (5 per eye).

## Why you'd pick it

Real things people build with this class of library:

- A **reading attention tracker** — the app knows which word your eyes are on, not just which part of the screen you're aiming at. (This is what Windows81's private `com.eyegaze` reader does.)
- An **accessibility gaze cursor** — a head- or eye-driven pointer for someone who can't use a touchscreen.
- An **eyewear or gaming example** — glance-based input, e.g. look left to dodge.
- An **in-class fatigue alert** — detect drooping eyelids and prompt a break.

Pick this fork when you want the prebuilt artifacts and the Android packaging already done, and you don't want a Flutter plugin in your project. If you want the Flutter plugin, use cornpip's repository — this fork changes nothing about the library itself.

## How to use it

No command line in this section. Everything happens inside Android Studio.

### From the C side — the public API

You never call these directly from Kotlin unless you write Path B; the JNI bridge calls them for you. One-line meaning for each:

- `mp_face_mesh_create` — start a context from a model file; returns the handle everything else uses.
- `mp_face_mesh_process_nv21` — feed one NV21 camera frame to the context; returns landmarks.
- `mp_face_mesh_is_tracking` — true while the internal region of interest stays locked on the face.
- `mp_face_mesh_release_result` — free the result struct once you are done inspecting it.
- `mp_face_mesh_last_error` — what went wrong (string; empty when nothing did).
- `mp_face_mesh_destroy` — release the context when done.

JNI bridge methods used by the reader app (Path A):

- `nativeCreate` — one-time bootstrap; calls `mp_face_mesh_create` for you.
- `nativeFeatureExtractNv21` — one NV21 frame in, 9-float features out; NaN-filled when no face is found.
- `nativeDestroy` — release the context lifecycle.

Minimal Kotlin sketch (Path A):

```kotlin
val handle = EyegazeFaceMeshNative.nativeCreate(attentionModelPath)
val features = EyegazeFaceMeshNative.nativeFeatureExtractNv21(
    handle, yBuf, vuBuf, width, height, yRowStride, vuRowStride, rotationDegrees)
// features: FloatArray(9), or NaN entries when no face was found
EyegazeFaceMeshNative.nativeDestroy(handle)
```

1. Download the latest release zip and unzip it.
2. In the **Project** view, switch to **Project** layout. Drag the `.so` files into `app/src/main/jniLibs/arm64-v8a/` (create the folders if they don't exist). Use `x86_64/` instead if you run on an emulator.
3. Copy `assets/models/face_landmark_with_attention.tflite` and `assets/models/face_detection_full_range.tflite` from the zip into `app/src/main/assets/` (drag-and-drop works). The `.tflite` files are the compiled neural networks — you are not changing them.
4. Android Studio packs all of this into the final app when you press Run; you don't need to look at aapt or the build scripts.
5. Load the libraries once, early (e.g. in your `Application` class or a companion `init` block):

```kotlin
System.loadLibrary("mpfm_tensorflowlite_c")
System.loadLibrary("mediapipe_face_mesh")
System.loadLibrary("eygaze_mesh_jni")   // only if you use the JNI bridge
```

Then choose a path.

**Path A — the JNI bridge (fast, opinionated).** Windows81's reader apps use this. Write a tiny class named `EyegazeFaceMeshNative` in your own package whose `external fun` declarations match `src/eygaze_mesh_jni.cc`:

```kotlin
class EyegazeFaceMeshNative {
    external fun nativeCreate(modelPath: String): Long
    external fun nativeDestroy(handle: Long)
    external fun nativeFeatureExtractNv21(
        handle: Long, yBuf: ByteArray, vuBuf: ByteArray,
        width: Int, height: Int, yRowStride: Int, vuRowStride: Int,
        rotationDegrees: Int): FloatArray
    companion object {
        init { System.loadLibrary("eygaze_mesh_jni") }
    }
}
```

Create it with the asset path of the attention model, feed it NV21 frames from the camera (Android's native camera format), and you get back a `FloatArray` of 9 numbers per frame — the eye-region feature vector Windows81's app feeds to its gaze regressor. If you want your own calibration regressor, this is the shape it expects. On failure the array comes back filled with NaN, so check with `Float.isNaN`.

**Path B — the C API (lower level).** Call `mp_face_mesh_process_nv21(context, &nv21, nullptr, rotation, 0)` from your own JNI shim. You create the context with `mp_face_mesh_create` and receive a raw `MpFaceMeshResult` struct. More work, but every byte of the result is yours.

**What the output means, plainly.**

- `FloatArray(9)` — the eye-region feature vector. Nine numbers, same kind Windows81's gaze regressor consumes.
- 478 landmarks × 3 floats — per-landmark `(x, y, z)` in image space. Indices follow MediaPipe's 478-mesh layout (eye corners near 33 and 133; irides at 468 and 473). Do not assume beyond what upstream documents.
- No face found: Path A gives NaN floats; Path B gives `landmarks_count == 0`. Check every frame.

### Code examples

**Delegate selection** — one line picks the backend: `if (nativeEyegaze != null) loadNativeBackend() else loadTaskBackend()`. `nativeEyegaze` is the bridge object from Path A; `null` means the `.so` failed to load, so you fall back to the original task model. You see: native iris tracking when available, the old path otherwise.

```kotlin
val backend = if (nativeEyegaze != null) loadNativeBackend() else loadTaskBackend()
```

**Activity integration** — you call extraction in `onCreate`, after the camera preview exists:

```kotlin
class XYZActivity : AppCompatActivity() {
    override fun onCreate(s: Bundle?) {
        super.onCreate(s)
        setContentView(R.layout.activity_xyz)
        cameraPreview.start { frame ->
            val f = nativeEyegaze?.extract(frame) // FloatArray(9), or NaN on no face
            gazeOverlay.update(f)
        }
    }
}
```

You see: eyes tracked as soon as frames flow; no native setup beyond the `loadLibrary` calls above.

**Feature vector reader** — prints each number with its name:

```kotlin
fun printFeatures(v: FloatArray) {
    val names = listOf("aEyeX","aEyeY","aEyeZ","bEyeX","bEyeY","bEyeZ","aIrisX","aIrisY","bIrisX","bIrisY").take(9)
    v.forEachIndexed { i, x -> println("${names[i]} = $x") }
}
```

You see nine labeled lines, e.g. `aEyeX = 0.42`.

## Honest limits

- Realistic capture sizes are 64×480 or 640×480. The phone's front-camera software pipeline sets that ceiling, not this library.
- 60 FPS is the ideal; 18–25 fps is realistic.
- Eye-distance estimation works at roughly 25–45 cm. Closer or farther is unreliable.
- This arrangement has not been tested for production anything — no soak testing, no crash reporting, no device matrix beyond the Surface Duo it was built for. Treat it as a working build, not a hardened product.

## If you want to build the library yourself

This is not the normal path; the release zip is. You will need the Android NDK r26d and CMake, and you are on your own with the exact flags — see `.github/workflows/main.yml` for the working recipe (NDK r26d 26.3.11579264, arm64-v8a and x86_64 targets, `-Wl,-z,max-page-size=16384` preserved from upstream for Android 15).

## License and attribution

Apache-2.0 / BSD-3-Clause. Cornpip's notices are preserved. The underlying C++, the TFLite models, and the license notices are unchanged from upstream.

Inkling 2 Small Free (OpenCode agent, model `opencode/inkling-2-small-free`) assembled this arrangement under Windows81's direction. This is agentic engineering — AI-authored under human direction, not pure human authorship. It is not an authoritative upstream; for the Flutter plugin and upstream issues, use [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh).
