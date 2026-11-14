# mediapipe_face_mesh — Android native build fork

**Provenance.** This is a fork of [cornpip/mediapipe_face_mesh](https://github.com/cornpip/mediapipe_face_mesh)
(upstream commit 962bddab0ad1951401b40553189dfbcf3d3473e7), Apache-2.0 / BSD-3-Clause.
It is **assembled by Inkling 2 Small Free** (the model driving the build-session) as a
build/branching arrangement to produce Android native libraries via GitHub Actions.
It is **not** an authoritative upstream: for the Flutter plugin itself, use cornpip's
repository. License texts are preserved (LICENSE, LICENSE-APACHE-2.0.txt, NOTICES,
THIRD_PARTY_NOTICES.md).

## What the workflow builds

`.github/workflows/build-android.yml` builds, per ABI (`arm64-v8a`, `x86_64`):

- `libmediapipe_face_mesh.so` — compiled from `src/` via `android/cmake/CMakeLists.txt`
- `libmpfm_tensorflowlite_c.so` — copied verbatim from `android/src/main/jniLibs/<abi>/` (prebuilt, shipped upstream)

On tag pushes (`v*`) it attaches `mediapipe-facemesh-android.zip` to a GitHub Release.

## Local reproduction

Requires Android SDK + NDK r26d (26.3.11579264) and CMake:

    sdkmanager "ndk;26.3.11579264" "cmake;3.22.1"
    export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.3.11579264"
    cmake -S android/cmake -B build/arm64-v8a \
      -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Release
    cmake --build build/arm64-v8a
