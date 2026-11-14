# mediapipe-facemesh-action — Android build of MediaPipe's face+iris library

> **Credit.** Forked arrangement of [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh) (upstream commit `962bddab`), Apache-2.0 / BSD-3-Clause. Assembly by **Inkling 2 Small Free** (OpenCode agent, model `opencode/inkling-2-small-free`). This is **agentic engineering** — AI-authored under human direction, not pure human authorship. Not an authoritative upstream — for the Flutter plugin, use cornpip's repository.
>
> If you have used a terminal to `git clone`, `cmake`, or `make` before, you can use this. No computer-vision background needed.

---

## The purpose of this fork, in one sentence

An Android app, called `com.eyegaze`, is being built for **private use** by its author. That is the whole reason this fork exists. No product, no distribution, no sales channel.

## What a reader will want to know

**Why is it public at all?** Because the account that hosts this repository publishes the work that gets done on every session — this public record is documentation, not a promotion. The "product" angle literally does not exist.

**What does the app do?** Measures the position of a person's eyes on a phone screen, so that whoever built it can know which word on a page is being fixated on — not which region of the screen the face is pointed at. That is finer than anything a phone camera can deliver by default, and the existing stock face-mesh model (Google's `face_landmarker.task`) is not sufficient for that level of precision.

**Why does this matter to anyone else?** It probably does not, directly. If you reached this repository looking for an off-the-shelf library for something similar, what you actually want is [`cornpip/mediapipe_face_mesh`](https://github.com/cornpip/mediapipe_face_mesh) — this fork changes none of that library's behavior. This fork exists so that a specific private app, on a specific Surface Duo, can build against the native library without pulling in a Flutter plugin it has no use for. That is the entire delta.

**Is it maintained?** Care provided will be purely of the build shape (CI, README, release cadence, JNI bridge). The underlying C++, the TFLite models, and the license notices are unchanged from upstream.

**Can I reuse it?** Yes. The two `.so` files plus the TFLite models bundled in the release are what make any of this interesting; everything else is plumbing.

The library answers two questions, fast, from a camera frame:

1. **"Where is the face?"** Roughly: which pixels belong to the face and where are the eyes, nose, mouth.
2. **"Where is the pupil?"** 5 points per eye, with sub-pixel precision.

If you want to build anything that consumes `libmediapipe_face_mesh.so`, you have everything you need right here — and you do not need to care about the `com.eyegaze` context at all. This fork's value for you is the prebuilt artifacts and the build arrangement.

## What changed vs upstream

Upstream (`cornpip/mediapipe_face_mesh`) is a Flutter plugin. This fork **does not** change any of the upstream C++, the TFLite models, or the JNI ABI — it repackages the native build so it can be consumed from a plain Android app.

| Area | What changed | Why |
|---|---|---|
| Build driver | Added `.github/workflows/main.yml` (CMake + NDK) | Flutter tooling is unavailable in this project |
| Release cadence | Release zips with ISO-8601 UTC tag from the commit | Every commit on `main` produces an artifact |
| README | Written for command-line readers, not CV engineers | Anyone here can follow it |
| Upstream source | **Unchanged** | The plugin and the C API are identical |
| Upstream license notices | **Kept** | Apache-2.0 / BSD-3-Clause is honored |
| Attribution | Credits Inkling 2 Small Free, model `opencode/inkling-2-small-free` | Agentic engineering, declared explicitly |

If any of those rows ever starts to look wrong, the honest move is to revert the fork and use upstream directly — this fork is a convenience, not a replacement.

## A 60-second glossary

- **Face landmark** — a specific named point on the face. For example, "landmark 33" is the outer corner of the left eye. The library tracks 468 of them.
- **Iris landmark** — five points on each iris (center + four boundary points). Where this library differs from a plain "face landmark" model, which has no iris points.
- **TFLite model** — the file the runtime actually loads. Think of it as a compiled neural-network binary. The model files ship in `src/main/assets/models/`.
- **.so file** — a compiled shared library for Android. You can `dlopen()` it, or link against it at build time.
- **ABI** — processor architecture. The build produces two: `arm64-v8a` (64-bit ARM; most modern phones) and `x86_64` (emulators and some x86 tablets).
- **NDK** — the Android Native Development Kit. A toolchain that produces `.so` files for Android. This repo builds via the NDK, not via Flutter or Gradle.
- **CMake** — a build generator that drives the NDK.

If any of those are still unclear, open an issue — the guide can be reworded. This is aimed at command-line users, not at CV engineers.

---

## What the workflow does (plain-English walkthrough)

The file `.github/workflows/build.yml` (or `build-android.yml`) does three things whenever you push to `main`:

1. Sets up the Android NDK r26d and CMake on a clean runner.
2. Compiles `src/mediapipe_face_mesh.cc` plus its helpers with `arm64-v8a` and `x86_64` targets.
3. Packages the resulting `.so` files plus the prebuilt TFLite runtime, and uploads them as a single `mediapipe-facemesh-android.zip`.

Then a second job attaches that zip to a GitHub Release with an ISO-8601 UTC timestamp in the tag/name, e.g. `v20261114T024700Z` — compact, sortable, generated from the commit's own timestamp, not the wall clock.

**What is in the zip**, per ABI:

- `libmediapipe_face_mesh.so` — the library itself
- `libmpfm_tensorflowlite_c.so` — the TFLite runtime (prebuilt, copied verbatim)

**What you need to consume it**: the TFLite runtime (already included), a camera, and whatever shape of app you are building. There is no SDK lock-in; if your app can load a `.so` and call a C function, it works.

---

## Using it from an Android app

1. Download the latest release artifact (arm64-v8a unless you are on an emulator).
2. Place the two `.so` files under your app's `jniLibs/arm64-v8a/`.
3. In your app's CMakeLists (or via `System.loadLibrary`), link against `mediapipe_face_mesh`.
4. Call `mediapipe_face_mesh_process_*` with the camera frame; read output landmarks.

A complete example is the upstream cornpip Flutter plugin, whose call sites are trivially portable to plain C/Java.

---

## Local reproduction

Requires Android SDK + NDK r26d (26.3.11579264) and CMake:

    sdkmanager "ndk;26.3.11579264" "cmake;3.22.1"
    export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.3.11579264"
    cmake -S android/cmake -B build/arm64-v8a \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release
    cmake --build build/arm64-v8a

Output: `build/arm64-v8a/libmediapipe_face_mesh.so`.

## Limitations

- Only `arm64-v8a` and `x86_64`.
- Requires Android 7.0+ (SDK 24).
- 16 KB page-size link flag (`-Wl,-z,max-page-size=16384`) is preserved from upstream for Android 15 compatibility.
- The bundled TFLite runtime (`libmpfm_tensorflowlite_c.so`) is matched by SONAME; do not replace it with a generic TFLite without verifying symbol compatibility.

## License and attribution

Apache-2.0 / BSD-3-Clause. Cornpip's notices are preserved.

This arrangement was assembled via OpenCode with the `opencode/inkling-2-small-free` model (Inkling 2 Small Free). It's agentic engineering — AI-authored under human direction, not pure human authorship. It is not an authoritative upstream; for the Flutter plugin and upstream issues, use [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh).
