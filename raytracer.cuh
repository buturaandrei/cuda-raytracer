/**
 * raytracer.cuh - Funzioni Core del Ray Tracing
 * 
 * Questo file contiene l'ALGORITMO principale del ray tracing:
 * 
 * COME FUNZIONA IL RAY TRACING:
 * ============================
 * 
 * 1. Per ogni pixel dello schermo:
 *    a. Calcola la direzione del raggio dalla camera attraverso quel pixel
 *    b. Lancia il raggio nella scena
 *    c. Trova l'oggetto più vicino che il raggio colpisce
 *    d. Calcola il colore in quel punto (illuminazione + materiale)
 *    e. Se il materiale è riflettente, rimbalza il raggio e ripeti
 * 
 * 2. Il colore finale viene scritto nel pixel corrispondente
 * 
 * 
 * SCHEMA DEL PROCESSO:
 * ===================
 * 
 *     Camera
 *        \
 *         \  Raggio primario
 *          \
 *           *-----> Oggetto colpito
 *          /|\
 *         / | \
 *        /  |  \
 *    Shadow |  Reflection
 *     Ray   |    Ray
 *           |
 *        Normale
 */

#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "materials.cuh"
#include "hit_record.cuh"
#include "sphere.cuh"
#include "triangle.cuh"
#include "bvh.cuh"
#include "light.cuh"
#include <float.h>

// Numero massimo di rimbalzi per riflessioni
#ifndef MAX_DEPTH
#define MAX_DEPTH 5
#endif

// Numero di sfere nella scena
#ifndef NUM_SPHERES
#define NUM_SPHERES 7
#endif

// Numero di triangoli (definito esternamente quando si usa il mesh)
#ifndef NUM_TRIANGLES
#define NUM_TRIANGLES 0
#endif

/**
 * Trova l'oggetto più vicino colpito dal raggio
 * 
 * @param ray - Il raggio da testare
 * @param t_min - Distanza minima (evita self-intersection)
 * @param t_max - Distanza massima
 * @param spheres - Array delle sfere nella scena
 * @return HitRecord con le informazioni dell'intersezione (o hit=false se nulla)
 * 
 * ALGORITMO:
 * 1. Inizia con distanza "closest" = t_max (molto lontano)
 * 2. Per ogni sfera nella scena:
 *    - Se il raggio la interseca a distanza t < closest:
 *      - Aggiorna closest = t
 *      - Salva le informazioni di questa intersezione
 * 3. Alla fine, abbiamo l'oggetto PIÙ VICINO
 * 
 * Perché il più vicino? Perché è quello che VEDIAMO!
 * Gli oggetti dietro sono nascosti.
 */
__host__ __device__ inline HitRecord findClosestHit(
    const Ray& ray, 
    float t_min, 
    float t_max, 
    Sphere* spheres
) {
    HitRecord rec;
    float closest = t_max;

    for (int i = 0; i < NUM_SPHERES; i++) {
        float t;
        if (spheres[i].intersect(ray, t_min, closest, t)) {
            closest = t;
            rec.hit = true;
            rec.t = t;
            rec.point = ray.at(t);
            rec.normal = spheres[i].getNormal(rec.point);
            rec.material = spheres[i].material;
        }
    }
    return rec;
}

/**
 * Test veloce raggio-AABB (Axis-Aligned Bounding Box)
 * Slab method - molto efficiente!
 * Restituisce true se il raggio POTREBBE colpire qualcosa nel box.
 */
__host__ __device__ inline bool rayIntersectsAABB(
    const Ray& ray,
    const Vec3& boxMin,
    const Vec3& boxMax,
    float t_max
) {
    float tmin = 0.001f;
    float tmax = t_max;
    
    // Test su ogni asse (X, Y, Z)
    for (int i = 0; i < 3; i++) {
        float origin = (i == 0) ? ray.origin.x : ((i == 1) ? ray.origin.y : ray.origin.z);
        float dir = (i == 0) ? ray.direction.x : ((i == 1) ? ray.direction.y : ray.direction.z);
        float bmin = (i == 0) ? boxMin.x : ((i == 1) ? boxMin.y : boxMin.z);
        float bmax = (i == 0) ? boxMax.x : ((i == 1) ? boxMax.y : boxMax.z);
        
        if (fabsf(dir) < 0.0001f) {
            // Raggio parallelo a questo asse
            if (origin < bmin || origin > bmax) return false;
        } else {
            float invD = 1.0f / dir;
            float t0 = (bmin - origin) * invD;
            float t1 = (bmax - origin) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            tmin = (t0 > tmin) ? t0 : tmin;
            tmax = (t1 < tmax) ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
    }
    return true;
}

/**
 * Trova l'oggetto più vicino colpito dal raggio (versione con triangoli)
 * 
 * Questa versione testa sia sfere che triangoli.
 * 
 * @param ray - Il raggio da testare
 * @param t_min - Distanza minima
 * @param t_max - Distanza massima
 * @param spheres - Array delle sfere
 * @param triangles - Array dei triangoli (può essere NULL)
 * @param numTriangles - Numero di triangoli da testare
 * @return HitRecord con le informazioni dell'intersezione
 * 
 * PERFORMANCE NOTE:
 * ================
 * Con molti triangoli (>1000), questo diventa lento perché
 * testiamo TUTTI i triangoli per ogni raggio.
 * Per scene grandi si usa un BVH (Bounding Volume Hierarchy)
 * per saltare velocemente i triangoli lontani.
 */
__host__ __device__ inline HitRecord findClosestHitWithTriangles(
    const Ray& ray, 
    float t_min, 
    float t_max, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    int numTriangles,
    Vec3 meshBoundsMin,
    Vec3 meshBoundsMax
) {
    HitRecord rec;
    float closest = t_max;

    // Test sfere
    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (spheres[i].intersect(ray, t_min, closest, t)) {
            closest = t;
            rec.hit = true;
            rec.t = t;
            rec.point = ray.at(t);
            rec.normal = spheres[i].getNormal(rec.point);
            rec.material = spheres[i].material;
        }
    }
    
    // Test triangoli - SOLO se il raggio colpisce il bounding box!
    if (triangles != nullptr && numTriangles > 0) {
        // Early reject: se il raggio non colpisce il bounding box, salta TUTTI i triangoli
        if (rayIntersectsAABB(ray, meshBoundsMin, meshBoundsMax, closest)) {
            for (int i = 0; i < numTriangles; i++) {
                float t;
                if (triangles[i].intersect(ray, t_min, closest, t)) {
                    closest = t;
                    rec.hit = true;
                    rec.t = t;
                    rec.point = ray.at(t);
                    rec.normal = triangles[i].getNormal(rec.point);
                    rec.material = triangles[i].material;
                }
            }
        }
    }
    
    return rec;
}

/**
 * Trova l'oggetto più vicino usando BVH per i triangoli (VELOCE!)
 * 
 * Questa versione usa il BVH grid per accelerare la ricerca.
 * Invece di testare tutti i triangoli, testa solo quelli nelle celle
 * attraversate dal raggio.
 */
__host__ __device__ inline HitRecord findClosestHitWithBVH(
    const Ray& ray, 
    float t_min, 
    float t_max, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    const BVHGrid& bvh,
    int* bvhTriangleIndices
) {
    HitRecord rec;
    float closest = t_max;

    // Test sfere (poche, brute force va bene)
    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (spheres[i].intersect(ray, t_min, closest, t)) {
            closest = t;
            rec.hit = true;
            rec.t = t;
            rec.point = ray.at(t);
            rec.normal = spheres[i].getNormal(rec.point);
            rec.material = spheres[i].material;
        }
    }
    
    // Test triangoli con BVH!
    if (triangles != nullptr && bvhTriangleIndices != nullptr) {
        float t;
        int triIndex;
        if (intersectBVHGrid(ray, bvh, triangles, bvhTriangleIndices, t_min, closest, t, triIndex)) {
            if (t < closest) {
                closest = t;
                rec.hit = true;
                rec.t = t;
                rec.point = ray.at(t);
                rec.normal = triangles[triIndex].getNormal(rec.point);
                rec.material = triangles[triIndex].material;
            }
        }
    }
    
    return rec;
}

/**
 * Controlla se un punto è in ombra usando BVH per i triangoli
 * 
 * OTTIMIZZATO: usa BVH per triangoli quindi è veloce anche con migliaia di triangoli!
 */
__host__ __device__ inline bool isInShadowWithBVH(
    const Vec3& point, 
    const Vec3& lightDir, 
    float lightDist, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    const BVHGrid& bvh,
    int* bvhTriangleIndices
) {
    Ray shadowRay(point + lightDir * 0.001f, lightDir);
    
    // Test sfere (poche, brute-force)
    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (spheres[i].intersect(shadowRay, 0.001f, lightDist, t)) return true;
    }
    
    // Test triangoli con BVH (veloce!)
    if (triangles != nullptr && bvhTriangleIndices != nullptr) {
        float t;
        int triIndex;
        if (intersectBVHGrid(shadowRay, bvh, triangles, bvhTriangleIndices, 
                             0.001f, lightDist, t, triIndex)) {
            return true;
        }
    }
    
    return false;
}

/**
 * Controlla se un punto è in ombra rispetto a una luce
 * 
 * @param point - Il punto sulla superficie da testare
 * @param lightDir - Direzione verso la luce (normalizzata)
 * @param lightDist - Distanza dalla luce
 * @param spheres - Array delle sfere nella scena
 * @return true se c'è un oggetto tra il punto e la luce (ombra)
 * 
 * FUNZIONAMENTO:
 * 1. Crea un "shadow ray" dal punto verso la luce
 * 2. Se questo raggio colpisce qualcosa PRIMA di raggiungere la luce:
 *    → Il punto è in ombra
 * 3. Altrimenti:
 *    → Il punto è illuminato
 * 
 * Il piccolo offset (0.001f) serve a evitare che il punto
 * "colpisca se stesso" (shadow acne).
 */
__host__ __device__ inline bool isInShadow(
    const Vec3& point, 
    const Vec3& lightDir, 
    float lightDist, 
    Sphere* spheres
) {
    Ray shadowRay(point + lightDir * 0.001f, lightDir);
    for (int i = 0; i < NUM_SPHERES; i++) {
        float t;
        if (spheres[i].intersect(shadowRay, 0.001f, lightDist, t)) return true;
    }
    return false;
}

/**
 * Controlla se un punto è in ombra (versione ottimizzata)
 * NOTA: Per performance, testiamo solo le SFERE per le ombre!
 * Testare 6800 triangoli per ogni shadow ray sarebbe troppo lento.
 */
__host__ __device__ inline bool isInShadowWithTriangles(
    const Vec3& point, 
    const Vec3& lightDir, 
    float lightDist, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    int numTriangles
) {
    Ray shadowRay(point + lightDir * 0.001f, lightDir);
    
    // Test SOLO sfere per le ombre (i triangoli sarebbero troppo lenti)
    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (spheres[i].intersect(shadowRay, 0.001f, lightDist, t)) return true;
    }
    
    // SKIP triangoli per performance!
    // Per ombre accurate sui triangoli servirebbe un BVH
    
    return false;
}

/**
 * FUNZIONE PRINCIPALE: Traccia un raggio e calcola il colore
 * 
 * Questa è la funzione più importante del raytracer!
 * 
 * @param ray - Il raggio da tracciare
 * @param maxDepth - Massimo numero di rimbalzi (per riflessioni)
 * @param spheres - Array delle sfere nella scena
 * @param lights - Array delle luci nella scena
 * @return Colore finale per questo raggio (RGB)
 * 
 * ALGORITMO DETTAGLIATO:
 * =====================
 * 
 * 1. LOOP per ogni "rimbalzo" (bounces):
 * 
 *    2. Trova l'oggetto più vicino (findClosestHit)
 * 
 *    3. Se non colpisce nulla → restituisci il colore del CIELO
 *       (Usiamo un gradiente basato sulla direzione Y del raggio)
 * 
 *    4. Se colpisce qualcosa:
 *       a. Calcola l'ILLUMINAZIONE DIRETTA:
 *          - Per ogni luce nella scena:
 *            - Controlla se il punto è in ombra
 *            - Se illuminato: calcola diffuse + specular
 *       
 *       b. Aggiungi componente AMBIENT (luce di base)
 *       
 *       c. Se il materiale è RIFLETTENTE:
 *          - Calcola la direzione riflessa
 *          - Crea un nuovo raggio per il prossimo bounce
 *          - Moltiplica "throughput" per reflectivity
 *          
 *       d. Se NON riflettente: stop, abbiamo finito
 * 
 * THROUGHPUT:
 * ===========
 * È un fattore che tiene traccia di quanta luce "passa attraverso"
 * ogni rimbalzo. Ogni volta che il raggio rimbalza, la luce si
 * attenua secondo il colore e la reflectivity del materiale.
 */
__host__ __device__ inline Vec3 traceRay(
    Ray ray, 
    int maxDepth, 
    Sphere* spheres, 
    Light* lights
) {
    Vec3 finalColor(0, 0, 0);
    Vec3 throughput(1, 1, 1);

    for (int depth = 0; depth < maxDepth; depth++) {
        HitRecord hit = findClosestHit(ray, 0.001f, FLT_MAX, spheres);

        // Se non colpisce nulla → colore del cielo + sole
        if (!hit.hit) {
            float t = 0.5f * (ray.direction.y + 1.0f);
            t = fmaxf(0.0f, fminf(1.0f, t));
            Vec3 skyBottom(0.8f, 0.85f, 0.95f);
            Vec3 skyTop(0.3f, 0.5f, 0.9f);
            Vec3 sky = skyBottom * (1.0f - t) + skyTop * t;
            
            // === SOLE NEL CIELO ===
            Vec3 sunDir = Vec3(5, 10, 5).normalize();
            float sunDot = ray.direction.dot(sunDir);
            if (sunDot > 0) {
                float sunDisc = powf(sunDot, 256.0f);
                float sunGlow = powf(sunDot, 8.0f) * 0.3f;
                Vec3 sunColor = Vec3(1.0f, 0.95f, 0.8f);
                sky = sky + sunColor * (sunDisc + sunGlow);
            }
            
            finalColor = finalColor + throughput * sky;
            break;
        }

        // Materiale emissivo (sorgente di luce)
        if (hit.material.type == EMISSIVE) {
            finalColor = finalColor + throughput * hit.material.color * hit.material.emission;
            break;
        }
        
        // === PAVIMENTO A SCACCHIERA ===
        Vec3 matColor = hit.material.color;
        if (hit.normal.y > 0.9f) {
            int checkX = (int)floorf(hit.point.x);
            int checkZ = (int)floorf(hit.point.z);
            bool isWhite = ((checkX + checkZ) % 2 == 0);
            if (hit.point.x < 0) isWhite = !isWhite;
            if (hit.point.z < 0) isWhite = !isWhite;
            matColor = isWhite ? Vec3(0.9f, 0.88f, 0.85f) : Vec3(0.15f, 0.15f, 0.18f);
        }

        // Calcola illuminazione diretta
        Vec3 directLight(0, 0, 0);
        
        // Specular intensity dipende dal materiale:
        // - REFLECTIVE: specular forte (superficie lucida)
        // - DIFFUSE: specular debole (superficie opaca)
        float specIntensity = (hit.material.type == REFLECTIVE) ? 
                              (0.5f + hit.material.reflectivity * 0.5f) : 0.1f;
        float specPower = (hit.material.type == REFLECTIVE) ? 64.0f : 16.0f;
        
        for (int i = 0; i < 2; i++) {
            Vec3 lightDir = lights[i].position - hit.point;
            float lightDist = lightDir.length();
            lightDir = lightDir.normalize();

            if (!isInShadow(hit.point, lightDir, lightDist, spheres)) {
                // Diffuse (Lambert)
                float diff = fmaxf(0.0f, hit.normal.dot(lightDir));
                
                // Specular (Blinn-Phong) - più forte per materiali riflettenti!
                Vec3 viewDir = (ray.origin - hit.point).normalize();
                Vec3 halfDir = (lightDir + viewDir).normalize();
                float spec = powf(fmaxf(0.0f, hit.normal.dot(halfDir)), specPower);
                
                // Attenuazione
                float atten = lights[i].intensity / (1.0f + 0.05f * lightDist);

                directLight = directLight + lights[i].color * matColor * diff * atten;
                directLight = directLight + lights[i].color * spec * atten * specIntensity;
            }
        }

        // Ambient + direct light
        Vec3 ambient = matColor * 0.1f;
        finalColor = finalColor + throughput * (directLight + ambient);

        // Riflessioni
        if (hit.material.type == REFLECTIVE && hit.material.reflectivity > 0.0f) {
            Vec3 reflDir = ray.direction.reflect(hit.normal);
            ray = Ray(hit.point + reflDir * 0.001f, reflDir);
            throughput = throughput * hit.material.color * hit.material.reflectivity;
        } else {
            break;
        }
    }

    return finalColor;
}

/**
 * FUNZIONE PRINCIPALE CON TRIANGOLI: Traccia un raggio nella scena completa
 * 
 * Versione estesa che supporta sia sfere che triangoli.
 * Usata quando si renderizza un modello 3D caricato da file OBJ.
 */
__host__ __device__ inline Vec3 traceRayWithTriangles(
    Ray ray, 
    int maxDepth, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    int numTriangles,
    Light* lights,
    int numLights,
    Vec3 meshBoundsMin,
    Vec3 meshBoundsMax
) {
    Vec3 finalColor(0, 0, 0);
    Vec3 throughput(1, 1, 1);

    for (int depth = 0; depth < maxDepth; depth++) {
        HitRecord hit = findClosestHitWithTriangles(ray, 0.001f, FLT_MAX, 
                                                     spheres, numSpheres,
                                                     triangles, numTriangles,
                                                     meshBoundsMin, meshBoundsMax);

        // Se non colpisce nulla → colore del cielo + sole
        if (!hit.hit) {
            float t = 0.5f * (ray.direction.y + 1.0f);
            t = fmaxf(0.0f, fminf(1.0f, t));
            Vec3 skyBottom(0.8f, 0.85f, 0.95f);
            Vec3 skyTop(0.3f, 0.5f, 0.9f);
            Vec3 sky = skyBottom * (1.0f - t) + skyTop * t;
            
            // === SOLE NEL CIELO ===
            Vec3 sunDir = Vec3(5, 10, 5).normalize();
            float sunDot = ray.direction.dot(sunDir);
            if (sunDot > 0) {
                float sunDisc = powf(sunDot, 256.0f);
                float sunGlow = powf(sunDot, 8.0f) * 0.3f;
                Vec3 sunColor = Vec3(1.0f, 0.95f, 0.8f);
                sky = sky + sunColor * (sunDisc + sunGlow);
            }
            
            finalColor = finalColor + throughput * sky;
            break;
        }

        // Materiale emissivo (sorgente di luce)
        if (hit.material.type == EMISSIVE) {
            finalColor = finalColor + throughput * hit.material.color * hit.material.emission;
            break;
        }

        // Correggi normali invertite (double-sided geometry)
        // Se la normale punta via dalla camera, invertila
        Vec3 normal = hit.normal;
        if (normal.dot(ray.direction) > 0) {
            normal = normal * -1.0f;  // Flip della normale
        }
        
        // === PAVIMENTO A SCACCHIERA ===
        Vec3 matColor = hit.material.color;
        if (hit.normal.y > 0.9f) {
            int checkX = (int)floorf(hit.point.x);
            int checkZ = (int)floorf(hit.point.z);
            bool isWhite = ((checkX + checkZ) % 2 == 0);
            if (hit.point.x < 0) isWhite = !isWhite;
            if (hit.point.z < 0) isWhite = !isWhite;
            matColor = isWhite ? Vec3(0.9f, 0.88f, 0.85f) : Vec3(0.15f, 0.15f, 0.18f);
        }

        // Calcola illuminazione diretta
        Vec3 directLight(0, 0, 0);
        for (int i = 0; i < numLights; i++) {
            Vec3 lightDir = lights[i].position - hit.point;
            float lightDist = lightDir.length();
            lightDir = lightDir.normalize();

            if (!isInShadowWithTriangles(hit.point, lightDir, lightDist, 
                                          spheres, numSpheres, triangles, numTriangles)) {
                // Diffuse (Lambert) - usa normale corretta
                float diff = fmaxf(0.0f, normal.dot(lightDir));
                
                // Specular (Blinn-Phong)
                Vec3 viewDir = (ray.origin - hit.point).normalize();
                Vec3 halfDir = (lightDir + viewDir).normalize();
                float spec = powf(fmaxf(0.0f, normal.dot(halfDir)), 32.0f);
                
                // Attenuazione
                float atten = lights[i].intensity / (1.0f + 0.05f * lightDist);

                directLight = directLight + lights[i].color * matColor * diff * atten;
                directLight = directLight + lights[i].color * spec * atten * 0.3f;
            }
        }

        // Ambient + direct light
        Vec3 ambient = matColor * 0.15f;  // Ambient leggermente più alto per i modelli
        finalColor = finalColor + throughput * (directLight + ambient);

        // Riflessioni
        if (hit.material.type == REFLECTIVE && hit.material.reflectivity > 0.0f) {
            Vec3 reflDir = ray.direction.reflect(normal);  // Usa normale corretta
            ray = Ray(hit.point + reflDir * 0.001f, reflDir);
            
            // Fresnel effect
            Vec3 viewDir = (ray.origin - hit.point).normalize();
            float cosTheta = fmaxf(0.0f, normal.dot(viewDir));
            float fresnel = hit.material.reflectivity + (1.0f - hit.material.reflectivity) * powf(1.0f - cosTheta, 5.0f);
            fresnel = fminf(fresnel, 0.5f);
            
            throughput = throughput * hit.material.color * fresnel;
        } else {
            break;
        }
    }

    return finalColor;
}

/**
 * FUNZIONE PRINCIPALE CON BVH: Ray tracing accelerato!
 * 
 * Usa il BVH grid per testare solo i triangoli rilevanti.
 * MOLTO più veloce della versione brute-force.
 */
__host__ __device__ inline Vec3 traceRayWithBVH(
    Ray ray, 
    int maxDepth, 
    Sphere* spheres,
    int numSpheres,
    Triangle* triangles,
    const BVHGrid& bvh,
    int* bvhTriangleIndices,
    Light* lights,
    int numLights
) {
    Vec3 finalColor(0, 0, 0);
    Vec3 throughput(1, 1, 1);

    for (int depth = 0; depth < maxDepth; depth++) {
        HitRecord hit = findClosestHitWithBVH(ray, 0.001f, FLT_MAX, 
                                              spheres, numSpheres,
                                              triangles, bvh, bvhTriangleIndices);

        if (!hit.hit) {
            float t = 0.5f * (ray.direction.y + 1.0f);
            t = fmaxf(0.0f, fminf(1.0f, t));
            Vec3 skyBottom(0.8f, 0.85f, 0.95f);
            Vec3 skyTop(0.3f, 0.5f, 0.9f);
            Vec3 sky = skyBottom * (1.0f - t) + skyTop * t;
            
            // === SOLE NEL CIELO ===
            Vec3 sunDir = Vec3(5, 10, 5).normalize();
            float sunDot = ray.direction.dot(sunDir);
            if (sunDot > 0) {
                float sunDisc = powf(sunDot, 256.0f);
                float sunGlow = powf(sunDot, 8.0f) * 0.3f;
                Vec3 sunColor = Vec3(1.0f, 0.95f, 0.8f);
                sky = sky + sunColor * (sunDisc + sunGlow);
            }
            
            finalColor = finalColor + throughput * sky;
            break;
        }

        if (hit.material.type == EMISSIVE) {
            finalColor = finalColor + throughput * hit.material.color * hit.material.emission;
            break;
        }

        // Correggi normali invertite (double-sided geometry)
        // Se la normale punta via dalla camera, invertila
        Vec3 normal = hit.normal;
        if (normal.dot(ray.direction) > 0) {
            normal = normal * -1.0f;  // Flip della normale
        }
        
        // === PAVIMENTO A SCACCHIERA ===
        Vec3 matColor = hit.material.color;
        if (hit.normal.y > 0.9f) {
            int checkX = (int)floorf(hit.point.x);
            int checkZ = (int)floorf(hit.point.z);
            bool isWhite = ((checkX + checkZ) % 2 == 0);
            if (hit.point.x < 0) isWhite = !isWhite;
            if (hit.point.z < 0) isWhite = !isWhite;
            matColor = isWhite ? Vec3(0.9f, 0.88f, 0.85f) : Vec3(0.15f, 0.15f, 0.18f);
        }

        // Calcola illuminazione diretta CON OMBRE (usando BVH per triangoli!)
        Vec3 directLight(0, 0, 0);
        for (int i = 0; i < numLights; i++) {
            Vec3 toLight = lights[i].position - hit.point;
            float lightDist = toLight.length();
            Vec3 lightDir = toLight.normalize();
            
            // Shadow test con BVH (veloce anche per triangoli!)
            if (!isInShadowWithBVH(hit.point, lightDir, lightDist,
                                    spheres, numSpheres,
                                    triangles, bvh, bvhTriangleIndices)) {
                // Diffuse (Lambert) - usa normale corretta
                float diff = fmaxf(0.0f, normal.dot(lightDir));
                
                // Specular (Blinn-Phong)
                Vec3 viewDir = (ray.origin - hit.point).normalize();
                Vec3 halfDir = (lightDir + viewDir).normalize();
                float spec = powf(fmaxf(0.0f, normal.dot(halfDir)), 32.0f);
                
                // Attenuazione
                float atten = lights[i].intensity / (1.0f + 0.05f * lightDist);
                
                directLight = directLight + lights[i].color * matColor * diff * atten;
                directLight = directLight + lights[i].color * spec * atten * 0.3f;
            }
        }

        Vec3 ambient = matColor * 0.15f;
        finalColor = finalColor + throughput * (directLight + ambient);

        if (hit.material.type == REFLECTIVE && hit.material.reflectivity > 0.0f) {
            Vec3 reflDir = ray.direction.reflect(normal);  // Usa normale corretta
            
            // FRESNEL: più riflettente ai bordi, meno al centro
            // Questo è il comportamento naturale di plastica/vetro/acqua
            Vec3 viewDir = (ray.origin - hit.point).normalize();
            float cosTheta = fmaxf(0.0f, normal.dot(viewDir));  // Usa normale corretta
            // Schlick's approximation per Fresnel
            float fresnel = hit.material.reflectivity + (1.0f - hit.material.reflectivity) * powf(1.0f - cosTheta, 5.0f);
            // Limita l'effetto fresnel per non esagerare
            fresnel = fminf(fresnel, 0.5f);
            
            ray = Ray(hit.point + reflDir * 0.001f, reflDir);
            throughput = throughput * hit.material.color * fresnel;
        } else {
            break;
        }
    }

    return finalColor;
}

/**
 * TONE MAPPING E GAMMA CORRECTION
 * ===============================
 * 
 * Dopo aver calcolato il colore, dobbiamo convertirlo per la visualizzazione:
 * 
 * 1. TONE MAPPING (Reinhard):
 *    - I calcoli possono produrre valori > 1.0 (HDR)
 *    - Formula: color_out = color_in / (1 + color_in)
 *    - Comprime i valori alti mantenendo i dettagli
 * 
 * 2. GAMMA CORRECTION:
 *    - I monitor non hanno risposta lineare alla luce
 *    - Formula: color_out = pow(color_in, 1/2.2)
 *    - Rende i colori più naturali
 * 
 * Queste trasformazioni sono applicate nel kernel/renderCPU,
 * non in questa funzione.
 */

#endif // RAYTRACER_CUH
