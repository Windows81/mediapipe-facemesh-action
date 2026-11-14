// JNI glue for com.eyegaze.EyegazeFaceMeshNative.
//
// The MediaPipe "face_landmark_with_attention" model is a refinement stage,
// not a standalone detector: it expects a crop around a face, not the full
// camera frame. Feed it the whole frame and its face flag never clears the
// presence threshold, which is why every frame read all-NaN before this
// rewrite. The fix is the same two-stage pipeline MediaPipe uses on device:
// a BlazeFace full-range pass first (face_detection_full_range.tflite) and
// the attention landmark model on the detected face rect.
//
// Builds the 9-feature eye vector the app regressor uses, from the
// 478-point landmarks this library emits.
#include <jni.h>
#include <android/log.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "mediapipe_face.h"

namespace {

constexpr char kLogTag[] = "eyegaze";

struct EyegazeBridge {
    MpFaceMeshContext* mesh;
    MpFaceDetectorContext* det;
};

EyegazeBridge* asBridge(jlong h) {
    return reinterpret_cast<EyegazeBridge*>(h);
}

// indices, must match EyeFeature.java
enum {
    IRIS_A = 468, IRIS_B = 473,
    INNER_A = 133, OUTER_A = 33, TOP_A = 159, BOT_A = 145,
    INNER_B = 362, OUTER_B = 263, TOP_B = 386, BOT_B = 374,
    NOSE = 1,
    FEM_MIN_LANDMARKS = 468,
};

float eyeEar(const MpLandmark* lm, int innerI, int outerI, int topI, int botI) {
    const auto& i = lm[innerI]; const auto& o = lm[outerI];
    const auto& t = lm[topI];   const auto& b = lm[botI];
    float cy = (t.y + b.y) / 2.f;
    (void) cy;
    float w = std::fabs(o.x - i.x), h = std::fabs(b.y - t.y);
    if (w < 1e-6f || h < 1e-6f) return -1.f;
    float ear = h / w;
    return ear;
}

float eyeDxDySelectIris(const MpLandmark* lm, int irisI, int innerI, int outerI, int topI, int botI, int pick) {
    const auto& i = lm[innerI]; const auto& o = lm[outerI];
    const auto& t = lm[topI];   const auto& b = lm[botI];
    float cx = (i.x + o.x) / 2.f, cy = (t.y + b.y) / 2.f;
    float w = std::fabs(o.x - i.x), h = std::fabs(b.y - t.y);
    const auto& p = lm[irisI];
    return pick == 0 ? (p.x - cx) / w : (p.y - cy) / h;
}

bool extractFeatures(const MpLandmark* lm, int count, float* out9) {
    if (count < FEM_MIN_LANDMARKS) return false;
    float aDx = eyeDxDySelectIris(lm, IRIS_A, INNER_A, OUTER_A, TOP_A, BOT_A, 0);
    float aDy = eyeDxDySelectIris(lm, IRIS_A, INNER_A, OUTER_A, TOP_A, BOT_A, 1);
    float bDx = eyeDxDySelectIris(lm, IRIS_B, INNER_B, OUTER_B, TOP_B, BOT_B, 0);
    float bDy = eyeDxDySelectIris(lm, IRIS_B, INNER_B, OUTER_B, TOP_B, BOT_B, 1);
    float earA = eyeEar(lm, INNER_A, OUTER_A, TOP_A, BOT_A);
    float earB = eyeEar(lm, INNER_B, OUTER_B, TOP_B, BOT_B);
    if (earA < 0.12f || earB < 0.12f) return false;   // blink

    const auto& innerA = lm[INNER_A]; const auto& outerA = lm[OUTER_A];
    const auto& innerB = lm[INNER_B]; const auto& outerB = lm[OUTER_B];
    float io = std::fabs(outerB.x - outerA.x);
    if (io < 1e-6f) io = std::fabs(innerB.x - innerA.x);
    if (io < 1e-6f) return false;
    float cxA = (innerA.x + outerA.x) / 2.f, cxB = (innerB.x + outerB.x) / 2.f;
    const auto& tA = lm[TOP_A]; const auto& bA = lm[BOT_A];
    const auto& tB = lm[TOP_B]; const auto& bB = lm[BOT_B];
    float midY = ((tA.y + bA.y) / 2.f + (tB.y + bB.y) / 2.f) / 2.f;
    float midX = (cxA + cxB) / 2.f;
    const auto& nose = lm[NOSE];
    float nx = (nose.x - midX) / io;
    float ny = (nose.y - midY) / io;

    float wA = std::fabs(outerA.x - innerA.x), wB = std::fabs(outerB.x - innerB.x);
    float cAz = (innerA.z + outerA.z) / 2.f, cBz = (innerB.z + outerB.z) / 2.f;
    float zA = (lm[IRIS_A].z - cAz) / (wA < 1e-6f ? 1e-6f : wA);
    float zB = (lm[IRIS_B].z - cBz) / (wB < 1e-6f ? 1e-6f : wB);
    float izm = (zA + zB) / 2.f;
    float izd = zA - zB;
    float zNose = (nose.z - (cAz + cBz) / 2.f) / io;

    out9[0] = aDx; out9[1] = aDy; out9[2] = bDx; out9[3] = bDy;
    out9[4] = nx;  out9[5] = ny;  out9[6] = izm; out9[7] = izd; out9[8] = zNose;
    return true;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeCreate(
    JNIEnv* env, jclass, jstring jModelPath, jstring jDetectorPath) {
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
    const char* detectorPath = env->GetStringUTFChars(jDetectorPath, nullptr);

    MpFaceMeshCreateOptions opts{};
    opts.threads = 2;
    opts.min_detection_confidence = 0.5f;
    opts.min_tracking_confidence = 0.5f;
    opts.min_face_presence_confidence = 0.5f;
    opts.delegate = MP_DELEGATE_XNNPACK;
    opts.enable_roi_tracking = 1;
    opts.enable_iris = 1;
    opts.enable_attention_mesh = 1;        // face_landmark_with_attention.tflite
    opts.disable_delegate_fallback = 0;
    MpFaceMeshContext* mesh = mp_face_mesh_create(modelPath, &opts);
    if (!mesh) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "mesh model init failed: %s", mp_face_mesh_last_global_error());
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        env->ReleaseStringUTFChars(jDetectorPath, detectorPath);
        return 0;
    }

    MpFaceDetectorCreateOptions dopts{};
    dopts.threads = 2;
    dopts.min_detection_confidence = 0.5f;
    dopts.min_suppression_threshold = 0.3f;
    dopts.max_results = 1;
    dopts.delegate = MP_DELEGATE_XNNPACK;
    dopts.disable_delegate_fallback = 0;
    MpFaceDetectorContext* det = mp_face_detector_create(detectorPath, &dopts);
    if (!det) {
        // Without a detector there is no face rect to crop, and the attention
        // model would always read below presence confidence. Fail loudly so
        // the caller can log it, rather than ship a silently-broken bridge.
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
                            "detector model init failed: %s",
                            mp_face_detector_last_global_error());
        mp_face_mesh_destroy(mesh);
        env->ReleaseStringUTFChars(jModelPath, modelPath);
        env->ReleaseStringUTFChars(jDetectorPath, detectorPath);
        return 0;
    }
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    env->ReleaseStringUTFChars(jDetectorPath, detectorPath);
    return reinterpret_cast<jlong>(new EyegazeBridge{mesh, det});
}

JNIEXPORT void JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    EyegazeBridge* b = asBridge(handle);
    if (!b) return;
    if (b->mesh) mp_face_mesh_destroy(b->mesh);
    if (b->det) mp_face_detector_destroy(b->det);
    delete b;
}

// Returns a 9-float array, or an array of all NaNs when no valid mesh/face is found.
JNIEXPORT jfloatArray JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeFeatureExtractNv21(
    JNIEnv* env, jclass, jlong handle,
    jbyteArray yBuf, jbyteArray vuBuf,
    jint width, jint height,
    jint yRowStride, jint vuRowStride,
    jint rotationDegrees) {
    jfloatArray out = env->NewFloatArray(9);
    jfloat dummy[9] = { NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN };

    EyegazeBridge* bridge = asBridge(handle);
    if (!bridge) {
        env->SetFloatArrayRegion(out, 0, 9, dummy);
        return out;
    }
    jbyte* y = env->GetByteArrayElements(yBuf, nullptr);
    jbyte* vu = env->GetByteArrayElements(vuBuf, nullptr);
    if (!y || !vu) {
        env->ReleaseByteArrayElements(yBuf, y, JNI_ABORT);
        env->ReleaseByteArrayElements(vuBuf, vu, JNI_ABORT);
        env->SetFloatArrayRegion(out, 0, 9, dummy);
        return out;
    }
    MpNv21Image image{};
    image.y = reinterpret_cast<const uint8_t*>(y);
    image.vu = reinterpret_cast<const uint8_t*>(vu);
    image.width = width;
    image.height = height;
    image.y_bytes_per_row = yRowStride;
    image.vu_bytes_per_row = vuRowStride;

    // 1) Full-frame BlazeFace detection.
    MpFaceDetectorResult* detResult = mp_face_detector_process_nv21(
        bridge->det, &image, /*override_rect=*/ nullptr, rotationDegrees,
        /*mirror=*/ 0, /*roi_transform=*/ nullptr);
    MpNormalizedRect rect{};
    bool haveRect = false;
    if (detResult) {
        int best = -1;
        for (int32_t i = 0; i < detResult->detections_count; ++i) {
            if (best < 0 ||
                detResult->detections[i].score > detResult->detections[best].score) {
                best = i;
            }
        }
        if (best >= 0) {
            rect = detResult->detections[best].expanded_face_rect;
            haveRect = true;
        }
        mp_face_detector_release_result(detResult);
    } else {
        __android_log_print(ANDROID_LOG_WARN, kLogTag,
                            "detector pass failed: %s",
                            mp_face_detector_last_error(bridge->det));
    }

    bool ok = false;
    float features[9];
    if (haveRect) {
        // 2) Attention landmark model on the detected face crop.
        MpFaceMeshResult* result = mp_face_mesh_process_nv21(
            bridge->mesh, &image, &rect, rotationDegrees, /*mirror=*/ 0);
        ok = result && result->landmarks &&
             result->landmarks_count >= FEM_MIN_LANDMARKS;
        if (ok) ok = extractFeatures(result->landmarks, result->landmarks_count, features);
        mp_face_mesh_release_result(result);
    }

    env->ReleaseByteArrayElements(yBuf, y, JNI_ABORT);
    env->ReleaseByteArrayElements(vuBuf, vu, JNI_ABORT);

    const jfloat* src = ok ? features : dummy;
    env->SetFloatArrayRegion(out, 0, 9, src);
    return out;
}

JNIEXPORT jlong JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeLastErrorCode(JNIEnv*, jclass, jlong handle) {
    // No numeric code helper in either context; return 1 if either side
    // recorded an error, to flag "library reported a failure".
    EyegazeBridge* bridge = asBridge(handle);
    if (!bridge) return 0;
    const char* meshErr = bridge->mesh ? mp_face_mesh_last_error(bridge->mesh) : nullptr;
    if (meshErr && *meshErr != '\0') return 1L;
    const char* detErr = bridge->det ? mp_face_detector_last_error(bridge->det) : nullptr;
    if (detErr && *detErr != '\0') return 1L;
    return 0L;
}

JNIEXPORT jstring JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeLastError(JNIEnv* env, jclass, jlong handle) {
    EyegazeBridge* bridge = asBridge(handle);
    if (!bridge) return env->NewStringUTF("");
    const char* meshErr = bridge->mesh ? mp_face_mesh_last_error(bridge->mesh) : nullptr;
    if (!meshErr) meshErr = "";
    const char* detErr = bridge->det ? mp_face_detector_last_error(bridge->det) : nullptr;
    if (detErr && *detErr != '\0') {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "mesh=%s detector=%s", meshErr, detErr);
        return env->NewStringUTF(buf);
    }
    return env->NewStringUTF(meshErr);
}

}  // extern "C"
