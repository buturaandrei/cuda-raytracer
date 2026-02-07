/**
 * obj_loader.cuh - Caricatore File OBJ per CUDA Ray Tracing
 * ==========================================================
 * 
 * Il formato OBJ è uno standard de-facto per modelli 3D.
 * È un formato testuale semplice, leggibile dall'uomo.
 * 
 * STRUTTURA FILE OBJ
 * ==================
 * 
 * # Commento
 * v  x y z          # Vertice (posizione)
 * vt u v            # Texture coordinate
 * vn x y z          # Normale del vertice
 * f  v/vt/vn ...    # Faccia (indici di vertice/texture/normale)
 * 
 * IMPORTANTE: Gli indici in OBJ partono da 1, non da 0!
 * 
 * FACCE QUAD vs TRIANGOLI
 * =======================
 * I file OBJ possono avere facce con 3, 4 o più vertici.
 * Il ray tracing richiede triangoli, quindi:
 * 
 *   Quad (4 vertici):      Due triangoli:
 *   
 *     0---3                0---3    0
 *     |   |     -->        | / |   /|
 *     1---2                1    1---2
 * 
 * Ogni quad viene diviso in 2 triangoli: (0,1,2) e (0,2,3)
 */

#ifndef OBJ_LOADER_CUH
#define OBJ_LOADER_CUH

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include "vec3.cuh"
#include "triangle.cuh"

/**
 * Struttura per contenere i dati parsati dal file OBJ.
 * Viene usata sul HOST (CPU) per caricare il file.
 */
struct ObjMesh {
    std::vector<Vec3> vertices;    // Lista di tutti i vertici
    std::vector<Vec3> normals;     // Lista di tutte le normali
    std::vector<Triangle> triangles; // Triangoli risultanti
    
    // Bounding box per centrare/scalare il modello
    Vec3 minBound;
    Vec3 maxBound;
    Vec3 center;
    float scale;
    
    /**
     * Carica un file OBJ e converte tutto in triangoli.
     * 
     * @param filename  Percorso del file OBJ
     * @param material  Materiale da applicare a tutti i triangoli
     * @return          true se caricamento riuscito
     */
    bool load(const std::string& filename, Material material) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERRORE: Impossibile aprire " << filename << std::endl;
            return false;
        }
        
        std::cout << "Caricamento " << filename << "..." << std::endl;
        
        std::string line;
        int lineNum = 0;
        int faceCount = 0;
        
        // Inizializza bounding box
        minBound = Vec3(1e30f, 1e30f, 1e30f);
        maxBound = Vec3(-1e30f, -1e30f, -1e30f);
        
        while (std::getline(file, line)) {
            lineNum++;
            
            // Salta linee vuote e commenti
            if (line.empty() || line[0] == '#')
                continue;
            
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;
            
            if (prefix == "v") {
                // =====================
                // VERTICE: v x y z
                // =====================
                float x, y, z;
                iss >> x >> y >> z;
                Vec3 v(x, y, z);
                vertices.push_back(v);
                
                // Aggiorna bounding box
                minBound.x = fminf(minBound.x, x);
                minBound.y = fminf(minBound.y, y);
                minBound.z = fminf(minBound.z, z);
                maxBound.x = fmaxf(maxBound.x, x);
                maxBound.y = fmaxf(maxBound.y, y);
                maxBound.z = fmaxf(maxBound.z, z);
            }
            else if (prefix == "vn") {
                // =====================
                // NORMALE: vn x y z
                // =====================
                float x, y, z;
                iss >> x >> y >> z;
                normals.push_back(Vec3(x, y, z).normalize());
            }
            else if (prefix == "f") {
                // =====================
                // FACCIA: f v/vt/vn ...
                // =====================
                faceCount++;
                std::vector<int> faceVertices;
                std::vector<int> faceNormals;
                
                std::string token;
                while (iss >> token) {
                    // Parse "v/vt/vn" or "v//vn" or "v"
                    int vi = 0, vti = 0, vni = 0;
                    
                    // Trova le posizioni dei /
                    size_t pos1 = token.find('/');
                    if (pos1 == std::string::npos) {
                        // Formato: v
                        vi = std::stoi(token);
                    } else {
                        // Formato: v/... 
                        vi = std::stoi(token.substr(0, pos1));
                        
                        size_t pos2 = token.find('/', pos1 + 1);
                        if (pos2 != std::string::npos) {
                            // Formato: v/vt/vn o v//vn
                            std::string vtStr = token.substr(pos1 + 1, pos2 - pos1 - 1);
                            if (!vtStr.empty()) {
                                vti = std::stoi(vtStr);
                            }
                            std::string vnStr = token.substr(pos2 + 1);
                            if (!vnStr.empty()) {
                                vni = std::stoi(vnStr);
                            }
                        } else {
                            // Formato: v/vt
                            std::string vtStr = token.substr(pos1 + 1);
                            if (!vtStr.empty()) {
                                vti = std::stoi(vtStr);
                            }
                        }
                    }
                    
                    // IMPORTANTE: OBJ usa indici 1-based!
                    // Indici negativi sono relativi alla fine della lista
                    if (vi < 0) vi = vertices.size() + vi + 1;
                    if (vni < 0) vni = normals.size() + vni + 1;
                    
                    faceVertices.push_back(vi - 1);  // Converti a 0-based
                    faceNormals.push_back(vni > 0 ? vni - 1 : -1);
                }
                
                // =====================
                // TRIANGOLAZIONE
                // =====================
                // Converte poligoni (3, 4 o più vertici) in triangoli
                // usando la tecnica "fan triangulation"
                //
                //   0-----3        Triangoli:
                //   |\    |        (0,1,2)
                //   | \   |        (0,2,3)
                //   |  \  |
                //   1----2
                
                if (faceVertices.size() >= 3) {
                    for (size_t i = 1; i < faceVertices.size() - 1; i++) {
                        int i0 = faceVertices[0];
                        int i1 = faceVertices[i];
                        int i2 = faceVertices[i + 1];
                        
                        // Verifica che gli indici siano validi
                        if (i0 >= 0 && i0 < (int)vertices.size() &&
                            i1 >= 0 && i1 < (int)vertices.size() &&
                            i2 >= 0 && i2 < (int)vertices.size()) {
                            
                            // Calcola la normale del triangolo
                            Vec3 normal;
                            
                            // Usa normale dal file se disponibile
                            int n0 = faceNormals[0];
                            int n1 = faceNormals[i];
                            int n2 = faceNormals[i + 1];
                            
                            if (n0 >= 0 && n0 < (int)normals.size()) {
                                // Media delle normali dei vertici
                                normal = normals[n0];
                                if (n1 >= 0 && n1 < (int)normals.size())
                                    normal = normal + normals[n1];
                                if (n2 >= 0 && n2 < (int)normals.size())
                                    normal = normal + normals[n2];
                                normal = normal.normalize();
                            } else {
                                // Calcola normale dalla geometria
                                Vec3 edge1 = vertices[i1] - vertices[i0];
                                Vec3 edge2 = vertices[i2] - vertices[i0];
                                normal = edge1.cross(edge2).normalize();
                            }
                            
                            triangles.push_back(Triangle(
                                vertices[i0],
                                vertices[i1],
                                vertices[i2],
                                normal,
                                material
                            ));
                        }
                    }
                }
            }
        }
        
        file.close();
        
        // Calcola centro e scala
        center = (minBound + maxBound) * 0.5f;
        Vec3 size = maxBound - minBound;
        scale = fmaxf(fmaxf(size.x, size.y), size.z);
        
        std::cout << "Caricamento completato!" << std::endl;
        std::cout << "  Vertici:   " << vertices.size() << std::endl;
        std::cout << "  Normali:   " << normals.size() << std::endl;
        std::cout << "  Facce:     " << faceCount << std::endl;
        std::cout << "  Triangoli: " << triangles.size() << std::endl;
        std::cout << "  Bounding Box: (" << minBound.x << ", " << minBound.y << ", " << minBound.z << ")" << std::endl;
        std::cout << "             -> (" << maxBound.x << ", " << maxBound.y << ", " << maxBound.z << ")" << std::endl;
        std::cout << "  Centro:    (" << center.x << ", " << center.y << ", " << center.z << ")" << std::endl;
        std::cout << "  Scala:     " << scale << std::endl;
        
        return true;
    }
    
    /**
     * Trasforma tutti i triangoli per centrare e scalare il modello.
     * 
     * @param newCenter  Nuovo centro del modello
     * @param newScale   Nuova scala (dimensione massima)
     */
    void transformToFit(Vec3 newCenter, float newScale) {
        float scaleFactor = newScale / scale;
        
        std::cout << "Trasformazione modello..." << std::endl;
        std::cout << "  Scala factor: " << scaleFactor << std::endl;
        
        for (Triangle& tri : triangles) {
            // Centra sul nuovo origine e scala
            tri.v0 = (tri.v0 - center) * scaleFactor + newCenter;
            tri.v1 = (tri.v1 - center) * scaleFactor + newCenter;
            tri.v2 = (tri.v2 - center) * scaleFactor + newCenter;
            
            // Ricalcola normale (necessario se c'è scaling non uniforme)
            Vec3 edge1 = tri.v1 - tri.v0;
            Vec3 edge2 = tri.v2 - tri.v0;
            tri.normal = edge1.cross(edge2).normalize();
        }
        
        // Aggiorna bounding box
        minBound = (minBound - center) * scaleFactor + newCenter;
        maxBound = (maxBound - center) * scaleFactor + newCenter;
        center = newCenter;
        scale = newScale;
    }
};

#endif // OBJ_LOADER_CUH
