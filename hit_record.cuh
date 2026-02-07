/**
 * hit_record.cuh - Informazioni di Intersezione
 */

#ifndef HIT_RECORD_CUH
#define HIT_RECORD_CUH

#include "vec3.cuh"
#include "materials.cuh"
#include <float.h>

struct HitRecord {
    Vec3 point;
    Vec3 normal;
    float t;
    Material material;
    bool hit;

    __host__ __device__ HitRecord() : hit(false), t(FLT_MAX) {}
};

#endif // HIT_RECORD_CUH
