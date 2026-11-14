// JNI glue for com.eyegaze.EyegazeFaceMeshNative.
// Builds the 9-feature eye vector the app regressor uses, from the
// 478-point landmarks this library emits.
#include <jni.h>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "mediapipe_face.h"

namespace {

MpFaceMeshContext* asCtx(jlong h) {
    return reinterpret_cast<MpFaceMeshContext*>(h);
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
    float cx = (i.x + o.x) / 2.f, cy = (t.y + b.y) / 2.f;
    float w = std::fabs(o.x - i.x), h = std::fabs(b.y - t.y);
    if (w < 1e-6f || h < 1e-6f) return -1.f;
    float* earOut = nullptr; (void) earOut;
    float ear = h / w;
    (void) cx; (void) cy;
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
    JNIEnv* env, jclass, jstring jModelPath) {
    const char* modelPath = env->GetStringUTFChars(jModelPath, nullptr);
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
    MpFaceMeshContext* ctx = mp_face_mesh_create(modelPath, &opts);
    env->ReleaseStringUTFChars(jModelPath, modelPath);
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    if (handle) mp_face_mesh_destroy(asCtx(handle));
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

    if (handle == 0) {
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

    MpFaceMeshResult* result = mp_face_mesh_process_nv21(
        asCtx(handle), &image, nullptr /*override_rect*/, rotationDegrees, /*mirror=*/ 0);

    bool ok = result && result->landmarks && result->landmarks_count >= FEM_MIN_LANDMARKS;
    float features[9];
    if (ok) ok = extractFeatures(result->landmarks, result->landmarks_count, features);
    mp_face_mesh_release_result(result);
    env->ReleaseByteArrayElements(yBuf, y, JNI_ABORT);
    env->ReleaseByteArrayElements(vuBuf, vu, JNI_ABORT);

    const jfloat* src = ok ? features : dummy;
    env->SetFloatArrayRegion(out, 0, 9, src);
    return out;
}

JNIEXPORT const char* JNICALL
Java_com_eyegaze_EyegazeFaceMeshNative_nativeLastError(JNIEnv* env, jclass, jlong handle) {
    return mp_face_mesh_last_error(0);   // placeholder; may be unused
}

}  // extern "C"
