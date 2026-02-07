/**
 * materials.cuh - Sistema di Materiali per Ray Tracing
 */

#ifndef MATERIALS_CUH
#define MATERIALS_CUH

#include "vec3.cuh"

enum MaterialType { DIFFUSE, REFLECTIVE, EMISSIVE };

struct Material {
    Vec3 color;
    MaterialType type;
    float reflectivity;
    float emission;

    __host__ __device__ Material() 
        : color(Vec3(0.5f, 0.5f, 0.5f)), type(DIFFUSE), reflectivity(0), emission(0) {}
    
    __host__ __device__ Material(Vec3 c, MaterialType t, float r = 0, float e = 0)
        : color(c), type(t), reflectivity(r), emission(e) {}
};

#endif // MATERIALS_CUH
