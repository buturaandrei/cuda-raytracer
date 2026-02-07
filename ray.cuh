/**
 * ray.cuh - Raggio per Ray Tracing
 */

#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d.normalize()) {}
    __host__ __device__ Vec3 at(float t) const { return origin + direction * t; }
};

#endif // RAY_CUH
