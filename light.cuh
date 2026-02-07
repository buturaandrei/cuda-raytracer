/**
 * light.cuh - Sistema di Illuminazione
 */

#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "vec3.cuh"

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;

    __host__ __device__ Light() {}
    __host__ __device__ Light(Vec3 p, Vec3 c, float i) : position(p), color(c), intensity(i) {}
};

#endif // LIGHT_CUH
