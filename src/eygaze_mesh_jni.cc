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

// One source. Distinct builds per feature vector length from the same code;
// mirror is a runtime parameter (nativeFeatureExtractNv21 mirror argument).
// The 9-float default is a drop-in for existing callers; the 15-float build
// adds eye aspect ratios, eye/io widths and iris radii for the Surface Duo.
// The caller decides whether the frame needs horizontally mirroring.
#if defined(EYGAZE_FEATURES_15)
#  define FEATURE_COUNT 15
#else
#  define FEATURE_COUNT 9
#endif

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

float irisRadius(const MpLandmark* lm, int irisStart, float normW) {
    // Order-agnostic iris radius: mean distance of the 5 iris points from
    // their own centroid, normalized by the eye width. Holds for whichever
    // index inside the iris group is the center vs the ring.
    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (int i = 0; i < 5; ++i) {
        cx += lm[irisStart + i].x;
        cy += lm[irisStart + i].y;
        cz += lm[irisStart + i].z;
    }
    cx /= 5.f; cy /= 5.f; cz /= 5.f;
    float r = 0.f;
    for (int i = 0; i < 5; ++i) {
        float dx = lm[irisStart + i].x - cx, dy = lm[irisStart + i].y - cy;
        r += std::sqrt(dx * dx + dy * dy);
    }
    return (r / 5.f) / (normW < 1e-6f ? 1e-6f : normW);
}

bool extractFeatures(const MpLandmark* lm, int count, float* out15) {
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
    float irisRA = irisRadius(lm, IRIS_A, wA);
    float irisRB = irisRadius(lm, IRIS_B, wB);
    float wAN = wA / io, wBN = wB / io;

    out15[0] = aDx; out15[1] = aDy; out15[2] = bDx; out15[3] = bDy;
    out15[4] = nx;  out15[5] = ny;  out15[6] = izm; out15[7] = izd; out15[8] = zNose;
    out15[9] = earA; out15[10] = earB;
    out15[11] = wAN; out15[12] = wBN;
    out15[13] = irisRA; out15[14] = irisRB;
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
    opts.min_face_presence_confidence = 0.3f;
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

// Returns a 15-float array, or an array of all NaNs when no valid mesh/face is found.
JNIEXPORT jfloatArray JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeFeatureExtractNv21(
    JNIEnv* env, jclass, jlong handle,
    jbyteArray yBuf, jbyteArray vuBuf,
    jint width, jint height,
    jint yRowStride, jint vuRowStride,
    jint rotationDegrees, jboolean jMirror) {
    jfloatArray out = env->NewFloatArray(FEATURE_COUNT);
    jfloat dummy[FEATURE_COUNT];
    for (int i = 0; i < FEATURE_COUNT; ++i) dummy[i] = NAN;

    EyegazeBridge* bridge = asBridge(handle);
    if (!bridge) {
        env->SetFloatArrayRegion(out, 0, FEATURE_COUNT, dummy);
        return out;
    }
    jbyte* y = env->GetByteArrayElements(yBuf, nullptr);
    jbyte* vu = env->GetByteArrayElements(vuBuf, nullptr);
    if (!y || !vu) {
        env->ReleaseByteArrayElements(yBuf, y, JNI_ABORT);
        env->ReleaseByteArrayElements(vuBuf, vu, JNI_ABORT);
        env->SetFloatArrayRegion(out, 0, FEATURE_COUNT, dummy);
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
    static int s_probe_counter = 0;
    const bool probe = (++s_probe_counter % 15) == 0;
    float detScore = -1.0f;

    MpFaceDetectorResult* detResult = mp_face_detector_process_nv21(
        bridge->det, &image, /*override_rect=*/ nullptr, rotationDegrees,
        /*mirror=*/ jMirror ? 1 : 0, /*roi_transform=*/ nullptr);
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
            detScore = detResult->detections[best].score;
            haveRect = true;
        }
        mp_face_detector_release_result(detResult);
    } else {
        __android_log_print(ANDROID_LOG_WARN, kLogTag,
                            "detector pass failed: %s",
                            mp_face_detector_last_error(bridge->det));
    }

    bool ok = false;
    float features[15] = { NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN };
    float meshScore = -1.0f;
    int meshLandmarks = 0;
    if (haveRect) {
        // 2) Attention landmark model on the detected face crop.
        MpFaceMeshResult* result = mp_face_mesh_process_nv21(
            bridge->mesh, &image, &rect, rotationDegrees, jMirror ? 1 : 0);
        if (result) {
            meshScore = result->score;
            meshLandmarks = result->landmarks_count;
            ok = result->landmarks &&
                 result->landmarks_count >= FEM_MIN_LANDMARKS;
        }
        if (result && ok) {
            ok = extractFeatures(result->landmarks, result->landmarks_count, features);
        }
        // Same 2 Hz probe: raw iris landmark coordinates (clamped already), for the
        // edges diagnostic. 468-472 = iris group A; 473-477 = iris group B.
        if (result && probe) {
            const MpLandmark* lm = result->landmarks;
            __android_log_print(ANDROID_LOG_INFO, kLogTag,
                "iris: 468=%.4f,%.4f 469=%.4f,%.4f 470=%.4f,%.4f 471=%.4f,%.4f 472=%.4f,%.4f "
                "| 473=%.4f,%.4f 474=%.4f,%.4f 475=%.4f,%.4f 476=%.4f,%.4f 477=%.4f,%.4f",
                lm[468].x, lm[468].y, lm[469].x, lm[469].y, lm[470].x, lm[470].y,
                lm[471].x, lm[471].y, lm[472].x, lm[472].y,
                lm[473].x, lm[473].y, lm[474].x, lm[474].y, lm[475].x, lm[475].y,
                lm[476].x, lm[476].y, lm[477].x, lm[477].y);
            __android_log_print(ANDROID_LOG_INFO, kLogTag,
                "geom: 33=%.4f,%.4f 133=%.4f,%.4f 159=%.4f,%.4f 145=%.4f,%.4f "
                "| 263=%.4f,%.4f 362=%.4f,%.4f 386=%.4f,%.4f 374=%.4f,%.4f "
                "| 1=%.4f,%.4f",
                lm[33].x, lm[33].y, lm[133].x, lm[133].y, lm[159].x, lm[159].y, lm[145].x, lm[145].y,
                lm[263].x, lm[263].y, lm[362].x, lm[362].y, lm[386].x, lm[386].y, lm[374].x, lm[374].y,
                lm[1].x, lm[1].y);
        }
        mp_face_mesh_release_result(result);
    }

    // Probe: one line every ~15th frame (~2 Hz at 30 fps), coordinates and
    // scores only - never pixel data.
    if (probe) {
        if (haveRect) {
            __android_log_print(ANDROID_LOG_INFO, kLogTag,
                "probe: det=%.3f rect=(x=%.3f,y=%.3f,w=%.3f,h=%.3f,rot=%.2f) "
                "mesh=%.3f landmarks=%d feat_ok=%d v=%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                detScore, rect.x_center, rect.y_center, rect.width, rect.height,
                rect.rotation, meshScore, meshLandmarks, ok ? 1 : 0,
                features[0], features[1], features[2], features[3], features[4],
                features[5], features[6], features[7], features[8], features[9],
                features[10], features[11], features[12], features[13], features[14]);
        } else {
            __android_log_print(ANDROID_LOG_INFO, kLogTag,
                "probe: det=none (no detection this frame)");
        }
    }

    env->ReleaseByteArrayElements(yBuf, y, JNI_ABORT);
    env->ReleaseByteArrayElements(vuBuf, vu, JNI_ABORT);

    const jfloat* src = ok ? features : dummy;
    env->SetFloatArrayRegion(out, 0, FEATURE_COUNT, src);
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
