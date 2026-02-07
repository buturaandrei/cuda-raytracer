/**
 * vec3.cuh - Vettore 3D per Ray Tracing
 * 
 * CUDA Header (.cuh) - contiene codice che deve funzionare sia su CPU che GPU
 * 
 * Il vettore Vec3 è la struttura fondamentale del ray tracer:
 * - Rappresenta POSIZIONI nello spazio 3D (x, y, z)
 * - Rappresenta DIREZIONI (vettori normalizzati)
 * - Rappresenta COLORI (r, g, b mappati su x, y, z)
 */

#ifndef VEC3_CUH
#define VEC3_CUH

#include "cuda_runtime.h"
#include <math.h>

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }

    __host__ __device__ float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    
    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    
    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
    
    __host__ __device__ Vec3 reflect(const Vec3& n) const {
        return *this - n * 2.0f * this->dot(n);
    }
};

#endif // VEC3_CUH
