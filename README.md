# mediapipe-facemesh-action — Android build of MediaPipe's face+iris library

> **Credit.** Fork of [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh) (upstream commit `962bddab`), Apache-2.0 / BSD-3-Clause. Assembled by **Inkling 2 Small Free** (OpenCode agent, model `opencode/inkling-2-small-free`) under Windows81's direction. AI-authored under human direction — not pure human authorship. For the Flutter plugin, use cornpip's repo.

Three files, one goal: fast eye-region tracking in a plain Android app.

- `libmediapipe_face_mesh.so` — the library that finds faces and computes landmarks (468 + 5 iris per eye).
- `libmpfm_tensorflowlite_c.so` — the TFLite runtime that actually runs the model.
- `libeygaze_mesh_jni.so` — a tiny JNI bridge from Java/Kotlin into the library.

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

- `nativeCreate(modelPath: String): Long` — start tracking. Returns a handle.
- `nativeFeatureExtractNv21(handle, yBuf, vuBuf, width, height, yRowStride, vuRowStride, rotationDegrees): FloatArray` — one NV21 camera frame in, 9-float eye-feature vector out (NaN when no face).
- `nativeDestroy(handle: Long)` — stop tracking.

**Plain C** (via your own JNI shim):

- `mp_face_mesh_create` — start a context.
- `mp_face_mesh_process_nv21` — feed NV21, get landmarks.
- `mp_face_mesh_release_result` — free the result.
- `mp_face_mesh_last_error` — what went wrong.
- `mp_face_mesh_destroy` — destroy the context.

## Minimal example

```kotlin
val handle = EyegazeFaceMeshNative.nativeCreate(filesDir.resolve("face_landmark_with_attention.tflite").absolutePath)
val features = EyegazeFaceMeshNative.nativeFeatureExtractNv21(
    handle, yBuf, vuBuf, width, height, yRowStride, vuRowStride, rotationDegrees)
EyegazeFaceMeshNative.nativeDestroy(handle)
```

`features` is 9 floats: the same feature vector used by Windows81's reader apps. Read them, feed them to your own regressor, done.

## Honest limits

- 640×480 or 64×480 camera frames only (front-camera software pipeline ceiling).
- 18–25 fps realistic.
- Eye-distance works at ~25–45 cm.
- Not production-hardened. Treat as a working build, not a hardened product.

## If you want to build the library yourself (not normal)

Release zips already ship the artifacts. The working recipe is `.github/workflows/main.yml` (NDK r26d, CMake, arm64-v8a + x86_64).

## License and attribution

Apache-2.0 / BSD-3-Clause. Not an authoritative upstream — for the Flutter plugin, see [cornpip's repository](https://github.com/cornpip/mediapipe_face_mesh).
