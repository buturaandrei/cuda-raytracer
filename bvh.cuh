/**
 * bvh.cuh - Bounding Volume Hierarchy (Grid-based)
 * =================================================
 * 
 * PROBLEMA:
 * =========
 * Con 6849 triangoli, ogni raggio deve testare TUTTI i triangoli = LENTO!
 * 
 * SOLUZIONE: BVH (Bounding Volume Hierarchy)
 * ==========================================
 * Dividiamo lo spazio 3D in una griglia di celle.
 * Ogni cella contiene solo i triangoli che la attraversano.
 * 
 * Quando un raggio attraversa la scena:
 * 1. Calcoliamo quali celle attraversa
 * 2. Testiamo SOLO i triangoli in quelle celle
 * 
 * ESEMPIO:
 * ========
 * Griglia 4x4x4 = 64 celle
 * 6849 triangoli distribuiti → ~107 triangoli/cella in media
 * Un raggio tipico attraversa ~4-8 celle
 * Test: 4-8 celle × ~107 triangoli = ~400-800 test invece di 6849!
 * 
 * SPEEDUP: ~10x !
 * 
 * 
 *     ┌───┬───┬───┬───┐
 *     │   │ ▲ │   │   │  ← Raggio attraversa solo 3 celle
 *     ├───┼─/─┼───┼───┤
 *     │  /│/  │   │   │
 *     ├─/─┼───┼───┼───┤
 *     │*  │   │   │   │  * = origine raggio
 *     └───┴───┴───┴───┘
 */

#ifndef BVH_CUH
#define BVH_CUH

#include "vec3.cuh"
#include "ray.cuh"
#include "triangle.cuh"
#include <vector>
#include <algorithm>

// Dimensione griglia BVH (8x8x8 = 512 celle)
#define BVH_GRID_SIZE 8
#define BVH_TOTAL_CELLS (BVH_GRID_SIZE * BVH_GRID_SIZE * BVH_GRID_SIZE)

/**
 * Struttura per una cella della griglia BVH.
 * Contiene gli indici dei triangoli che intersecano questa cella.
 */
struct BVHCell {
    int startIndex;     // Indice di partenza nell'array flat di indici
    int triangleCount;  // Numero di triangoli in questa cella
};

/**
 * Struttura BVH completa - usata su GPU
 */
struct BVHGrid {
    Vec3 gridMin;           // Minimo del bounding box della griglia
    Vec3 gridMax;           // Massimo del bounding box della griglia
    Vec3 cellSize;          // Dimensione di ogni cella
    Vec3 invCellSize;       // 1/cellSize per calcoli veloci
    
    BVHCell cells[BVH_TOTAL_CELLS];  // Array delle celle
    int* triangleIndices;   // Array flat degli indici triangoli (GPU)
    int totalIndices;       // Numero totale di indici
};

/**
 * Calcola l'indice della cella dalla posizione 3D
 */
__host__ __device__ inline int getCellIndex(int x, int y, int z) {
    return x + y * BVH_GRID_SIZE + z * BVH_GRID_SIZE * BVH_GRID_SIZE;
}

/**
 * Calcola le coordinate cella da una posizione nel mondo
 */
__host__ __device__ inline void worldToCell(
    const Vec3& pos, 
    const Vec3& gridMin, 
    const Vec3& invCellSize,
    int& cellX, int& cellY, int& cellZ
) {
    Vec3 local = pos - gridMin;
    cellX = (int)(local.x * invCellSize.x);
    cellY = (int)(local.y * invCellSize.y);
    cellZ = (int)(local.z * invCellSize.z);
    
    // Clamp agli estremi
    cellX = max(0, min(BVH_GRID_SIZE - 1, cellX));
    cellY = max(0, min(BVH_GRID_SIZE - 1, cellY));
    cellZ = max(0, min(BVH_GRID_SIZE - 1, cellZ));
}

/**
 * Test intersezione raggio con BVH grid.
 * Usa 3D-DDA (Digital Differential Analyzer) per attraversare le celle.
 */
__host__ __device__ inline bool intersectBVHGrid(
    const Ray& ray,
    const BVHGrid& bvh,
    Triangle* triangles,
    int* triangleIndices,
    float t_min,
    float t_max,
    float& out_t,
    int& out_triIndex
) {
    // Prima verifica se il raggio colpisce il bounding box globale
    float tmin = t_min;
    float tmax = t_max;
    
    // Test AABB del bounding box globale
    for (int i = 0; i < 3; i++) {
        float origin = (i == 0) ? ray.origin.x : ((i == 1) ? ray.origin.y : ray.origin.z);
        float dir = (i == 0) ? ray.direction.x : ((i == 1) ? ray.direction.y : ray.direction.z);
        float bmin = (i == 0) ? bvh.gridMin.x : ((i == 1) ? bvh.gridMin.y : bvh.gridMin.z);
        float bmax = (i == 0) ? bvh.gridMax.x : ((i == 1) ? bvh.gridMax.y : bvh.gridMax.z);
        
        if (fabsf(dir) < 0.0001f) {
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
    
    // Punto di ingresso nella griglia
    Vec3 entryPoint = ray.at(tmin + 0.001f);
    
    // Cella iniziale
    int cellX, cellY, cellZ;
    worldToCell(entryPoint, bvh.gridMin, bvh.invCellSize, cellX, cellY, cellZ);
    
    // Direzione di stepping (+1 o -1 per ogni asse)
    int stepX = (ray.direction.x >= 0) ? 1 : -1;
    int stepY = (ray.direction.y >= 0) ? 1 : -1;
    int stepZ = (ray.direction.z >= 0) ? 1 : -1;
    
    // Calcola tDelta (quanto t aumenta per attraversare una cella)
    float tDeltaX = (fabsf(ray.direction.x) > 0.0001f) ? 
                    fabsf(bvh.cellSize.x / ray.direction.x) : 1e30f;
    float tDeltaY = (fabsf(ray.direction.y) > 0.0001f) ? 
                    fabsf(bvh.cellSize.y / ray.direction.y) : 1e30f;
    float tDeltaZ = (fabsf(ray.direction.z) > 0.0001f) ? 
                    fabsf(bvh.cellSize.z / ray.direction.z) : 1e30f;
    
    // Calcola tMax (t al prossimo bordo cella)
    float cellMinX = bvh.gridMin.x + cellX * bvh.cellSize.x;
    float cellMinY = bvh.gridMin.y + cellY * bvh.cellSize.y;
    float cellMinZ = bvh.gridMin.z + cellZ * bvh.cellSize.z;
    
    float tMaxX, tMaxY, tMaxZ;
    if (fabsf(ray.direction.x) > 0.0001f) {
        float nextX = (stepX > 0) ? (cellMinX + bvh.cellSize.x) : cellMinX;
        tMaxX = (nextX - ray.origin.x) / ray.direction.x;
    } else {
        tMaxX = 1e30f;
    }
    if (fabsf(ray.direction.y) > 0.0001f) {
        float nextY = (stepY > 0) ? (cellMinY + bvh.cellSize.y) : cellMinY;
        tMaxY = (nextY - ray.origin.y) / ray.direction.y;
    } else {
        tMaxY = 1e30f;
    }
    if (fabsf(ray.direction.z) > 0.0001f) {
        float nextZ = (stepZ > 0) ? (cellMinZ + bvh.cellSize.z) : cellMinZ;
        tMaxZ = (nextZ - ray.origin.z) / ray.direction.z;
    } else {
        tMaxZ = 1e30f;
    }
    
    // Attraversa la griglia con DDA
    bool hitFound = false;
    float closestT = t_max;
    int closestTri = -1;
    
    int maxSteps = BVH_GRID_SIZE * 3;  // Limite di sicurezza
    
    for (int step = 0; step < maxSteps; step++) {
        // Controlla se siamo fuori dalla griglia
        if (cellX < 0 || cellX >= BVH_GRID_SIZE ||
            cellY < 0 || cellY >= BVH_GRID_SIZE ||
            cellZ < 0 || cellZ >= BVH_GRID_SIZE) {
            break;
        }
        
        // Ottieni la cella corrente
        int cellIdx = getCellIndex(cellX, cellY, cellZ);
        BVHCell cell = bvh.cells[cellIdx];
        
        // Testa tutti i triangoli in questa cella
        for (int i = 0; i < cell.triangleCount; i++) {
            int triIdx = triangleIndices[cell.startIndex + i];
            float t;
            if (triangles[triIdx].intersect(ray, t_min, closestT, t)) {
                if (t < closestT) {
                    closestT = t;
                    closestTri = triIdx;
                    hitFound = true;
                }
            }
        }
        
        // Se abbiamo trovato un hit in questa cella, e il t è minore
        // del prossimo bordo cella, possiamo fermarci
        float nextCellT = fminf(fminf(tMaxX, tMaxY), tMaxZ);
        if (hitFound && closestT < nextCellT) {
            break;
        }
        
        // Avanza alla prossima cella (DDA step)
        if (tMaxX < tMaxY && tMaxX < tMaxZ) {
            cellX += stepX;
            tMaxX += tDeltaX;
        } else if (tMaxY < tMaxZ) {
            cellY += stepY;
            tMaxY += tDeltaY;
        } else {
            cellZ += stepZ;
            tMaxZ += tDeltaZ;
        }
    }
    
    if (hitFound) {
        out_t = closestT;
        out_triIndex = closestTri;
        return true;
    }
    return false;
}

/**
 * Classe helper per costruire il BVH su CPU
 */
class BVHBuilder {
public:
    BVHGrid grid;
    std::vector<int> allIndices;  // Tutti gli indici triangoli (flat)
    std::vector<std::vector<int>> cellTriangles;  // Temporaneo per costruzione
    
    /**
     * Costruisce il BVH dai triangoli
     */
    void build(Triangle* triangles, int numTriangles, Vec3 boundsMin, Vec3 boundsMax) {
        // Espandi leggermente i bounds per evitare edge cases
        Vec3 padding = (boundsMax - boundsMin) * 0.01f;
        grid.gridMin = boundsMin - padding;
        grid.gridMax = boundsMax + padding;
        
        Vec3 gridSize = grid.gridMax - grid.gridMin;
        grid.cellSize = Vec3(
            gridSize.x / BVH_GRID_SIZE,
            gridSize.y / BVH_GRID_SIZE,
            gridSize.z / BVH_GRID_SIZE
        );
        grid.invCellSize = Vec3(
            1.0f / grid.cellSize.x,
            1.0f / grid.cellSize.y,
            1.0f / grid.cellSize.z
        );
        
        // Inizializza liste celle
        cellTriangles.resize(BVH_TOTAL_CELLS);
        for (auto& list : cellTriangles) {
            list.clear();
        }
        
        // Assegna ogni triangolo alle celle che interseca
        for (int i = 0; i < numTriangles; i++) {
            Triangle& tri = triangles[i];
            
            // Calcola bounding box del triangolo
            Vec3 triMin(
                fminf(fminf(tri.v0.x, tri.v1.x), tri.v2.x),
                fminf(fminf(tri.v0.y, tri.v1.y), tri.v2.y),
                fminf(fminf(tri.v0.z, tri.v1.z), tri.v2.z)
            );
            Vec3 triMax(
                fmaxf(fmaxf(tri.v0.x, tri.v1.x), tri.v2.x),
                fmaxf(fmaxf(tri.v0.y, tri.v1.y), tri.v2.y),
                fmaxf(fmaxf(tri.v0.z, tri.v1.z), tri.v2.z)
            );
            
            // Trova range di celle che il triangolo copre
            int minCellX, minCellY, minCellZ;
            int maxCellX, maxCellY, maxCellZ;
            worldToCell(triMin, grid.gridMin, grid.invCellSize, minCellX, minCellY, minCellZ);
            worldToCell(triMax, grid.gridMin, grid.invCellSize, maxCellX, maxCellY, maxCellZ);
            
            // Aggiungi triangolo a tutte le celle che copre
            for (int z = minCellZ; z <= maxCellZ; z++) {
                for (int y = minCellY; y <= maxCellY; y++) {
                    for (int x = minCellX; x <= maxCellX; x++) {
                        int cellIdx = getCellIndex(x, y, z);
                        cellTriangles[cellIdx].push_back(i);
                    }
                }
            }
        }
        
        // Costruisci array flat e compila BVHCell
        allIndices.clear();
        for (int i = 0; i < BVH_TOTAL_CELLS; i++) {
            grid.cells[i].startIndex = (int)allIndices.size();
            grid.cells[i].triangleCount = (int)cellTriangles[i].size();
            
            for (int idx : cellTriangles[i]) {
                allIndices.push_back(idx);
            }
        }
        
        grid.totalIndices = (int)allIndices.size();
        
        // Statistiche
        int maxTris = 0;
        int nonEmpty = 0;
        for (int i = 0; i < BVH_TOTAL_CELLS; i++) {
            if (grid.cells[i].triangleCount > 0) {
                nonEmpty++;
                maxTris = max(maxTris, grid.cells[i].triangleCount);
            }
        }
        
        printf("BVH costruito:\n");
        printf("  Griglia: %dx%dx%d = %d celle\n", BVH_GRID_SIZE, BVH_GRID_SIZE, BVH_GRID_SIZE, BVH_TOTAL_CELLS);
        printf("  Celle non vuote: %d\n", nonEmpty);
        printf("  Max triangoli/cella: %d\n", maxTris);
        printf("  Media triangoli/cella: %.1f\n", (float)numTriangles / nonEmpty);
        printf("  Totale riferimenti: %d\n", grid.totalIndices);
    }
};

#endif // BVH_CUH
