/**
 * kernel.cu - CUDA Ray Tracer - Main File
 * ========================================
 * 
 * Raytracer accelerato con GPU CUDA con visualizzazione in tempo reale.
 * 
 * 
 * ARCHITETTURA DELL'APPLICAZIONE
 * ==============================
 * 
 * Questo raytracer combina tre tecnologie:
 * 
 * 1. CUDA - Per il calcolo parallelo del ray tracing sulla GPU
 *    - Ogni pixel viene calcolato da un thread separato
 *    - Con 1280x720 pixel = 921,600 thread paralleli!
 * 
 * 2. OpenGL - Per visualizzare l'immagine sullo schermo
 *    - Usa una texture 2D aggiornata ogni frame
 *    - Disegna anche il debug overlay
 * 
 * 3. CUDA-OpenGL Interop - Il "ponte" tra CUDA e OpenGL
 *    - Permette a CUDA di scrivere direttamente nella memoria OpenGL
 *    - Evita costose copie GPU→CPU→GPU (zero-copy rendering)
 * 
 * 
 * FLUSSO DEI DATI (ogni frame)
 * ============================
 * 
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │                         GPU                                 │
 *   │  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
 *   │  │ CUDA Kernel  │ ──→  │     PBO      │ ──→  │  Texture  │ │
 *   │  │ (ray trace)  │      │ (buffer)     │      │  OpenGL   │ │
 *   │  └──────────────┘      └──────────────┘      └───────────┘ │
 *   └─────────────────────────────────────────────────────────────┘
 *                                                        │
 *                                                        ↓
 *                                                   ┌─────────┐
 *                                                   │ Schermo │
 *                                                   └─────────┘
 * 
 * PBO = Pixel Buffer Object: area di memoria GPU condivisa tra CUDA e OpenGL
 * 
 * 
 * ORGANIZZAZIONE DEI FILE:
 * ========================
 * 
 * vec3.cuh          - Vettore 3D con operazioni matematiche
 * ray.cuh           - Raggio (origine + direzione)
 * materials.cuh     - Materiali (diffuse, reflective, emissive)
 * hit_record.cuh    - Record di intersezione raggio-oggetto
 * sphere.cuh        - Geometria sfera con test intersezione
 * light.cuh         - Luce puntiforme
 * raytracer.cuh     - Algoritmo di ray tracing (traceRay, ecc.)
 * debug_overlay.cuh - Sistema di debug a schermo (font, drawString)
 * 
 * kernel.cu         - Questo file: main, OpenGL, input, kernel CUDA
 * 
 * 
 * INSTALLAZIONE DIPENDENZE:
 * =========================
 * In Visual Studio: Tasto destro progetto -> Manage NuGet Packages
 * Cerca e installa: "glfw" e "glew"
 * 
 * 
 * CONTROLLI:
 * ==========
 * - WASD:         Muovi camera
 * - Spazio/Shift: Sali/Scendi
 * - Mouse:        Guarda intorno (click sinistro per attivare)
 * - TAB:          Cambia CPU/CUDA rendering
 * - F1:           Toggle debug overlay
 * - R:            Reset posizione camera
 * - ESC:          Esci
 */

// ============================================================================
// INCLUDE - L'ordine è CRITICO!
// ============================================================================

/*
 * GLEW (GL Extension Wrangler) deve essere incluso PRIMA di qualsiasi
 * altro header OpenGL. GLEW carica dinamicamente le funzioni OpenGL
 * moderne che non sono disponibili nelle librerie di sistema.
 * 
 * Se includi gl.h prima di glew.h, ottieni errori di compilazione
 * perché glew.h ridefinisce alcune costanti.
 */
#include <GL/glew.h>

/*
 * GLFW (Graphics Library Framework) gestisce:
 * - Creazione finestra cross-platform (Windows, Linux, Mac)
 * - Contesto OpenGL
 * - Input (tastiera, mouse)
 * - Game loop e timing
 */
#include <GLFW/glfw3.h>

/*
 * Header CUDA:
 * - cuda_runtime.h: API CUDA (cudaMalloc, cudaMemcpy, kernel<<<>>>)
 * - device_launch_parameters.h: threadIdx, blockIdx, blockDim, gridDim
 * - cuda_gl_interop.h: funzioni per condividere memoria tra CUDA e OpenGL
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"  // Per cudaGraphicsGLRegisterBuffer, ecc.

// Standard C library
#include <stdio.h>   // printf, sprintf
#include <stdlib.h>  // malloc, free
#include <string.h>  // strcpy, memcpy
#include <float.h>   // FLT_MAX (massimo valore float)
#include <math.h>    // sinf, cosf, tanf, powf, sqrtf

// ============================================================================
// CONFIGURAZIONE - Definite PRIMA degli include dei nostri header!
// ============================================================================
/*
 * Queste macro devono essere definite PRIMA di includere i nostri header
 * perché alcuni di essi (come raytracer.cuh) le usano.
 * 
 * Esempio: raytracer.cuh usa NUM_SPHERES nel loop di ricerca sfere.
 */
#define IMAGE_WIDTH 1280     // Risoluzione HD (con BVH ora è veloce!)
#define IMAGE_HEIGHT 720
#define MAX_DEPTH 3          // Riflessi (ridotto da 4 per bilanciare performance)
                             // Valori più alti = riflessi più realistici ma più lenti
#define NUM_SPHERES 7        // Numero di sfere nella scena (array statico)

/*
 * I nostri header per il ray tracing.
 * Sono file .cuh (CUDA Header) invece di .h per indicare che contengono
 * codice che può essere eseguito sia su CPU che su GPU (__host__ __device__).
 */
#include "vec3.cuh"           // struct Vec3: vettore 3D con operazioni matematiche
#include "ray.cuh"            // struct Ray: raggio (origine + direzione)
#include "materials.cuh"      // enum MaterialType + struct Material
#include "hit_record.cuh"     // struct HitRecord: informazioni su dove il raggio colpisce
#include "sphere.cuh"         // struct Sphere: geometria sfera con intersect()
#include "triangle.cuh"       // struct Triangle: geometria triangolo per mesh 3D
#include "light.cuh"          // struct Light: luce puntiforme
#include "raytracer.cuh"      // Funzioni: findClosestHit, isInShadow, traceRay
#include "debug_overlay.cuh"  // GPUInfo, font5x7[], drawChar(), drawString()
#include "obj_loader.cuh"     // ObjMesh: caricatore file OBJ 3D

// ============================================================================
// VARIABILI GLOBALI - OpenGL
// ============================================================================
/*
 * GLFWwindow: la finestra dell'applicazione. GLFW gestisce automaticamente
 * la creazione della finestra su tutti i sistemi operativi.
 */
GLFWwindow* window = nullptr;

/*
 * PBO (Pixel Buffer Object): un buffer nella memoria GPU che può essere
 * condiviso tra CUDA e OpenGL. È il "ponte" che permette a CUDA di scrivere
 * direttamente nell'immagine che OpenGL mostrerà sullo schermo.
 * 
 * Senza PBO dovremmo fare: GPU(CUDA) → CPU → GPU(OpenGL) (molto lento!)
 * Con PBO facciamo:        GPU(CUDA) → GPU(OpenGL)        (zero-copy!)
 */
GLuint pbo = 0;

/*
 * textureID: la texture OpenGL che viene disegnata sullo schermo.
 * Viene aggiornata ogni frame con i dati del PBO.
 */
GLuint textureID = 0;

/*
 * cudaPBO: "handle" CUDA che rappresenta il PBO.
 * CUDA usa questo handle per mappare/smappare la memoria condivisa.
 */
cudaGraphicsResource* cudaPBO = nullptr;

// ============================================================================
// VARIABILI GLOBALI - Camera (sistema FPS-style)
// ============================================================================
/*
 * Sistema di controllo camera "First Person Shooter":
 * - Posizione: dove si trova la camera nello spazio 3D
 * - Yaw: rotazione orizzontale (sinistra/destra), 0° = guarda verso -Z
 * - Pitch: rotazione verticale (su/giù), limitato a ±89° per evitare gimbal lock
 * 
 *         Y (su)
 *         │
 *         │   Z (avanti nella convenzione OpenGL)
 *         │  /
 *         │ /
 *         │/_______ X (destra)
 *        O
 */
float camX = 0.0f, camY = 2.0f, camZ = 8.0f;  // Posizione iniziale: leggermente sopra e dietro l'origine
float camYaw = 0.0f, camPitch = 0.0f;          // Rotazione in gradi (non radianti!)
float moveSpeed = 5.0f;    // Unità al SECONDO (moltiplicato per deltaTime)
float mouseSpeed = 0.1f;   // Gradi per pixel di movimento mouse

/*
 * Gestione mouse:
 * - mouseCaptured: se true, il mouse è "catturato" (nascosto e infinito)
 * - firstMouse: evita il "salto" alla prima lettura del mouse
 * - lastMouseX/Y: posizione precedente per calcolare il delta
 */
bool mouseCaptured = false;
bool firstMouse = true;
double lastMouseX = 0, lastMouseY = 0;

// ============================================================================
// VARIABILI GLOBALI - Timing e FPS
// ============================================================================
/*
 * Delta time: tempo trascorso dall'ultimo frame.
 * Usato per rendere il movimento indipendente dal framerate.
 * Se non usassimo deltaTime, a 120 FPS ci muoveremmo 2x più velocemente che a 60 FPS!
 */
float deltaTime = 0.0f;
float lastFrame = 0.0f;    // Timestamp dell'ultimo frame (in secondi da avvio)

/*
 * Calcolo FPS:
 * Contiamo i frame per un secondo, poi aggiorniamo currentFPS e resettiamo.
 */
int frameCount = 0;
float fpsTimer = 0.0f;
int currentFPS = 0;

// ============================================================================
// VARIABILI GLOBALI - Modalità Rendering
// ============================================================================
bool useCPU = false;       // false = CUDA GPU (veloce), true = CPU (lento)
bool showDebugInfo = true; // Mostra pannello debug in alto a sinistra
bool showMesh = false;     // false = solo sfere (veloce), true = mostra modello 3D
bool enableReflections = false;  // false = solo illuminazione diretta (veloce), true = raytracing

// Informazioni sulla GPU per il debug overlay (struct definita in debug_overlay.cuh)
GPUInfo gpuInfo;

// ============================================================================
// MEMORIA GPU E CPU
// ============================================================================
/*
 * NAMING CONVENTION per puntatori:
 * - d_ = device (GPU memory, allocata con cudaMalloc)
 * - h_ = host (CPU memory, allocata con new/malloc)
 * 
 * Questa convenzione aiuta a capire immediatamente dove risiede la memoria
 * e previene errori come passare puntatori CPU a funzioni GPU.
 */
Sphere* d_spheres = nullptr;   // Array sfere su GPU (device)
Light* d_lights = nullptr;     // Array luci su GPU (device)
Triangle* d_triangles = nullptr; // Array triangoli su GPU (mesh 3D)
int numTriangles = 0;          // Numero di triangoli caricati dal file OBJ

// BVH per accelerare ray-triangle intersection
BVHGrid h_bvh;                 // BVH su CPU
int* d_bvhIndices = nullptr;   // Indici triangoli BVH su GPU
int* h_bvhIndices = nullptr;   // Indici triangoli BVH su CPU

// Bounding box del modello per early-reject dei raggi
Vec3 meshBoundsMin, meshBoundsMax;

uchar4* h_output = nullptr;    // Buffer immagine su CPU (per modalità CPU)
Sphere* h_spheres = nullptr;   // Array sfere su CPU (host)
Light* h_lights = nullptr;     // Luci su CPU (host)
Triangle* h_triangles = nullptr; // Array triangoli su CPU (host)

// ============================================================================
// KERNEL CUDA - IL CUORE DEL RAY TRACER!
// ============================================================================
/*
 * __global__ indica che questa funzione:
 * 1. Viene eseguita sulla GPU (device)
 * 2. Viene chiamata dalla CPU (host) con la sintassi <<<grid, block>>>
 * 3. Può chiamare funzioni __device__ ma NON funzioni normali CPU
 * 
 * PARALLELISMO CUDA:
 * ==================
 * Questo kernel viene eseguito da MILIONI di thread simultaneamente!
 * 
 * Organizzazione dei thread:
 * - Grid: griglia 2D di blocchi (80x45 blocchi per 1280x720)
 * - Block: gruppo di thread (16x16 = 256 thread per blocco)
 * - Thread: singola unità di esecuzione che processa 1 pixel
 * 
 *   Grid (80x45 blocchi)
 *   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
 *   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤ ← Ogni cella è un blocco 16x16
 *   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
 *   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
 * 
 *   Block (16x16 thread)
 *   ┌────────────────┐
 *   │ t t t t t t t t│
 *   │ t t t t t t t t│ ← Ogni 't' è un thread = 1 pixel
 *   │ t t t t t t t t│
 *   └────────────────┘
 * 
 * Le funzioni findClosestHit, isInShadow, traceRay sono in raytracer.cuh
 */
__global__ void renderKernel(uchar4* output, int width, int height,
                             Vec3 camPos, Vec3 camDir, Vec3 camUp,
                             Sphere* spheres, Light* lights) {
    /*
     * Calcola quale pixel questo thread deve processare.
     * 
     * - blockIdx: indice del blocco nella griglia (es. blocco 5,3)
     * - blockDim: dimensione del blocco (16,16)
     * - threadIdx: indice del thread nel blocco (es. thread 7,12)
     * 
     * Formula: coordinate_globale = blocco * dimensione_blocco + thread_locale
     */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Thread fuori dall'immagine? Esci subito.
    // (L'ultimo blocco potrebbe avere thread che "escono" dall'immagine)
    if (x >= width || y >= height) return;

    /*
     * SISTEMA DI COORDINATE DELLA CAMERA
     * ===================================
     * Costruiamo tre vettori ortogonali:
     * - forward: dove guarda la camera
     * - right: destra della camera (prodotto vettoriale forward × up)
     * - up: sopra della camera (prodotto vettoriale right × forward)
     * 
     *              up
     *              │
     *              │  forward
     *              │ /
     *              │/
     *    ─────────[*]────── right
     *            Camera
     */
    Vec3 forward = camDir;
    Vec3 right = forward.cross(camUp).normalize();  // cross product
    Vec3 up = right.cross(forward);  // ricalcola up per ortogonalità perfetta

    /*
     * PROIEZIONE PROSPETTICA
     * ======================
     * FOV (Field of View) = angolo di visione verticale.
     * 60° è un valore comune che simula la visione umana.
     * 
     *          ╱
     *         ╱
     *        ╱ ← FOV/2 = 30°
     *    [Camera]──────────────────
     *        ╲
     *         ╲
     *          ╲
     *      │←──── halfH = tan(30°) ────→│
     *             (a distanza 1)
     * 
     * aspect ratio = width/height = 16/9 per video HD
     * halfW = halfH * aspect (il viewport è più largo che alto)
     */
    float fov = 60.0f * 3.14159f / 180.0f;  // Converti gradi → radianti
    float aspect = (float)width / (float)height;
    float halfH = tanf(fov / 2.0f);  // Metà altezza del viewport a distanza 1
    float halfW = aspect * halfH;     // Metà larghezza del viewport

    /*
     * COORDINATE PIXEL → COORDINATE VIEWPORT
     * =======================================
     * Convertiamo le coordinate pixel (0,0)-(1279,719) in coordinate
     * viewport (-halfW,+halfH)-(+halfW,-halfH) centrate sull'origine.
     * 
     * Pixel:                    Viewport:
     * (0,0)──────(1279,0)       (-halfW,+halfH)────(+halfW,+halfH)
     *   │          │                   │                 │
     *   │          │         →         │       (0,0)     │
     *   │          │                   │                 │
     * (0,719)───(1279,719)     (-halfW,-halfH)────(+halfW,-halfH)
     * 
     * NOTA: +0.5f per centrare nel pixel (anti-aliasing povero)
     */
    float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * halfW;   // -halfW → +halfW
    float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * halfH;  // +halfH → -halfH

    /*
     * COSTRUZIONE DEL RAGGIO
     * ======================
     * Il raggio parte dalla camera e passa attraverso il punto (u,v)
     * sul piano virtuale a distanza 1 dalla camera.
     * 
     * Direzione = forward + right*u + up*v (combinazione lineare)
     * Normalizzata per avere lunghezza 1.
     */
    Vec3 rayDir = (forward + right * u + up * v).normalize();
    Ray ray(camPos, rayDir);

    /*
     * RAY TRACING!
     * ============
     * Questa singola chiamata fa tutto il lavoro:
     * - Trova intersezioni con le sfere
     * - Calcola illuminazione
     * - Gestisce riflessioni ricorsive
     * Vedi raytracer.cuh per i dettagli.
     */
    Vec3 color = traceRay(ray, MAX_DEPTH, spheres, lights);

    /*
     * TONE MAPPING (Reinhard Operator)
     * ================================
     * Il ray tracing produce colori HDR (High Dynamic Range) che possono
     * essere > 1.0 (es. luce molto intensa = 5.0).
     * 
     * I monitor mostrano solo colori 0-1 (LDR = Low Dynamic Range).
     * Il tone mapping comprime l'intervallo senza perdere i dettagli.
     * 
     * Formula Reinhard: Lout = Lin / (1 + Lin)
     * - Lin = 0.0 → Lout = 0.0
     * - Lin = 1.0 → Lout = 0.5
     * - Lin = ∞   → Lout = 1.0 (asintotico)
     */
    color.x = color.x / (1.0f + color.x);
    color.y = color.y / (1.0f + color.y);
    color.z = color.z / (1.0f + color.z);

    /*
     * GAMMA CORRECTION
     * ================
     * I monitor non mostrano i colori linearmente!
     * Un valore 0.5 non appare come "50% di luminosità".
     * 
     * Il monitor applica: display = input^2.2 (gamma)
     * Noi pre-compensiamo: output = color^(1/2.2)
     * Risultato: display = (color^(1/2.2))^2.2 = color ✓
     * 
     * Senza gamma correction l'immagine appare troppo scura!
     */
    color.x = powf(color.x, 1.0f / 2.2f);
    color.y = powf(color.y, 1.0f / 2.2f);
    color.z = powf(color.z, 1.0f / 2.2f);

    /*
     * SCRITTURA OUTPUT
     * ================
     * uchar4 = 4 byte (R, G, B, A) con valori 0-255.
     * fminf/fmaxf: clampa il valore in [0,1] prima di convertire.
     * 
     * Indice nel buffer lineare: y * width + x
     * (le immagini sono memorizzate riga per riga)
     */
    int idx = y * width + x;
    output[idx] = make_uchar4(
        (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),  // R
        (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),  // G
        (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),  // B
        255  // A (alpha = opaco)
    );
}

// ============================================================================
// KERNEL CUDA CON TRIANGOLI - Per modelli 3D caricati da file OBJ
// ============================================================================
/**
 * Versione estesa del kernel che supporta sia sfere che triangoli.
 * Usata per renderizzare mesh 3D (es. il diorama di Star Wars).
 * 
 * NOTA PERFORMANCE:
 * =================
 * Con ~6500 triangoli e senza BVH (Bounding Volume Hierarchy),
 * ogni raggio deve testare TUTTI i triangoli.
 * Questo è O(n) dove n = numero triangoli, quindi più lento.
 * Per scene grandi (>10000 triangoli) servirebbe un BVH.
 */
__global__ void renderKernelWithTriangles(
    uchar4* output, int width, int height,
    Vec3 camPos, Vec3 camDir, Vec3 camUp,
    Sphere* spheres, int numSpheres,
    Triangle* triangles, int numTriangles,
    Light* lights, int numLights,
    Vec3 meshBoundsMin, Vec3 meshBoundsMax
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    // Setup camera (identico al kernel base)
    Vec3 forward = camDir;
    Vec3 right = forward.cross(camUp).normalize();
    Vec3 up = right.cross(forward);

    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = (float)width / (float)height;
    float halfH = tanf(fov / 2.0f);
    float halfW = aspect * halfH;

    float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * halfW;
    float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * halfH;

    Vec3 rayDir = (forward + right * u + up * v).normalize();
    Ray ray(camPos, rayDir);

    // Ray tracing con supporto triangoli!
    Vec3 color = traceRayWithTriangles(ray, MAX_DEPTH, 
                                        spheres, numSpheres,
                                        triangles, numTriangles,
                                        lights, numLights,
                                        meshBoundsMin, meshBoundsMax);

    // Tone mapping (Reinhard)
    color.x = color.x / (1.0f + color.x);
    color.y = color.y / (1.0f + color.y);
    color.z = color.z / (1.0f + color.z);

    // Gamma correction
    color.x = powf(color.x, 1.0f / 2.2f);
    color.y = powf(color.y, 1.0f / 2.2f);
    color.z = powf(color.z, 1.0f / 2.2f);

    int idx = y * width + x;
    output[idx] = make_uchar4(
        (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),
        255
    );
}

// ============================================================================
// KERNEL GPU CON BVH - VERSIONE ACCELERATA
// ============================================================================
/*
 * Questo kernel usa il BVH (Bounding Volume Hierarchy) per accelerare
 * le intersezioni raggio-triangolo. Invece di testare tutti i ~7000 triangoli,
 * attraversa solo le celle della griglia che il raggio effettivamente tocca.
 * 
 * L'algoritmo 3D-DDA (Digital Differential Analyzer) avanza attraverso
 * la griglia voxel-by-voxel lungo il percorso del raggio.
 */
__global__ void renderKernelWithBVH(
    uchar4* output, int width, int height,
    Vec3 camPos, Vec3 camDir, Vec3 camUp,
    Sphere* spheres, int numSpheres,
    Triangle* triangles, int numTriangles,
    Light* lights, int numLights,
    BVHGrid bvh,  // Griglia BVH per accelerazione
    bool showMesh,  // Flag per mostrare/nascondere il modello 3D
    int rayDepth    // Profondità raggi (1 = no riflessioni, >1 = con riflessioni)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    // Setup camera
    Vec3 forward = camDir;
    Vec3 right = forward.cross(camUp).normalize();
    Vec3 up = right.cross(forward);

    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = (float)width / (float)height;
    float halfH = tanf(fov / 2.0f);
    float halfW = aspect * halfH;

    float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * halfW;
    float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * halfH;

    Vec3 rayDir = (forward + right * u + up * v).normalize();
    Ray ray(camPos, rayDir);

    // Ray tracing con BVH accelerato!
    // rayDepth controlla le riflessioni: 1 = solo luce diretta, MAX_DEPTH = con riflessioni
    Vec3 color = traceRayWithBVH(ray, rayDepth, 
                                  spheres, numSpheres,  // Sfere sempre visibili
                                  triangles,
                                  bvh,
                                  showMesh ? bvh.triangleIndices : nullptr,  // Mesh opzionale
                                  lights, numLights);

    // Tone mapping (Reinhard) - comprime HDR in [0,1]
    color.x = color.x / (1.0f + color.x);
    color.y = color.y / (1.0f + color.y);
    color.z = color.z / (1.0f + color.z);

    // Gamma correction (linear → sRGB)
    color.x = powf(color.x, 1.0f / 2.2f);
    color.y = powf(color.y, 1.0f / 2.2f);
    color.z = powf(color.z, 1.0f / 2.2f);

    int idx = y * width + x;
    output[idx] = make_uchar4(
        (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),
        255
    );
}

// ============================================================================
// RENDERING CPU (per confronto con CUDA)
// ============================================================================
/*
 * Questa funzione fa ESATTAMENTE lo stesso lavoro del kernel CUDA,
 * ma gira sulla CPU in modo sequenziale.
 * 
 * SCOPO DIDATTICO:
 * ================
 * Premendo TAB puoi confrontare le performance:
 * - CPU: processa 1 pixel alla volta, in sequenza (lento!)
 * - GPU: processa ~1 milione di pixel in parallelo (veloce!)
 * 
 * Su una scena tipica potresti vedere:
 * - CPU: 2-5 FPS
 * - GPU: 16-30 FPS (a seconda della gpu!)
 * 
 * Questo dimostra il potere del calcolo parallelo per problemi
 * "embarrassingly parallel" come il ray tracing, dove ogni pixel
 * è completamente indipendente dagli altri.
 */
void renderCPU(uchar4* output, int width, int height,
               Vec3 camPos, Vec3 camDir, Vec3 camUp,
               Sphere* spheres, Light* lights, int rayDepth) {
    
    // Stesso setup camera del kernel CUDA
    Vec3 forward = camDir;
    Vec3 right = forward.cross(camUp).normalize();
    Vec3 up = right.cross(forward);

    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = (float)width / (float)height;
    float halfH = tanf(fov / 2.0f);
    float halfW = aspect * halfH;

    // Loop sequenziale su tutti i pixel - MOLTO LENTO!
    // La GPU fa questo stesso lavoro ma con 1 milione di thread in parallelo
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Calcola coordinate viewport
            float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * halfW;
            float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * halfH;

            // Costruisci e traccia il raggio
            Vec3 rayDir = (forward + right * u + up * v).normalize();
            Ray ray(camPos, rayDir);
            Vec3 color = traceRay(ray, rayDepth, spheres, lights);

            // Tone mapping Reinhard
            color.x = color.x / (1.0f + color.x);
            color.y = color.y / (1.0f + color.y);
            color.z = color.z / (1.0f + color.z);

            // Gamma correction
            color.x = powf(color.x, 1.0f / 2.2f);
            color.y = powf(color.y, 1.0f / 2.2f);
            color.z = powf(color.z, 1.0f / 2.2f);

            // Scrivi pixel
            int idx = y * width + x;
            output[idx] = make_uchar4(
                (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),
                (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),
                (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),
                255
            );
        }
    }
}

/**
 * Rendering CPU con supporto triangoli.
 * ATTENZIONE: Con 6500+ triangoli questo sarà MOLTO lento!
 */
void renderCPUWithTriangles(uchar4* output, int width, int height,
                            Vec3 camPos, Vec3 camDir, Vec3 camUp,
                            Sphere* spheres, int numSpheres,
                            Triangle* triangles, int numTriangles,
                            Light* lights, int numLights,
                            Vec3 meshBoundsMin, Vec3 meshBoundsMax,
                            int rayDepth) {  // Aggiunto: profondità raggi
    
    Vec3 forward = camDir;
    Vec3 right = forward.cross(camUp).normalize();
    Vec3 up = right.cross(forward);

    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = (float)width / (float)height;
    float halfH = tanf(fov / 2.0f);
    float halfW = aspect * halfH;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * halfW;
            float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * halfH;

            Vec3 rayDir = (forward + right * u + up * v).normalize();
            Ray ray(camPos, rayDir);
            Vec3 color = traceRayWithTriangles(ray, rayDepth,  // Usa rayDepth invece di MAX_DEPTH
                                               spheres, numSpheres,
                                               triangles, numTriangles,
                                               lights, numLights,
                                               meshBoundsMin, meshBoundsMax);

            color.x = color.x / (1.0f + color.x);
            color.y = color.y / (1.0f + color.y);
            color.z = color.z / (1.0f + color.z);

            color.x = powf(color.x, 1.0f / 2.2f);
            color.y = powf(color.y, 1.0f / 2.2f);
            color.z = powf(color.z, 1.0f / 2.2f);

            int idx = y * width + x;
            output[idx] = make_uchar4(
                (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),
                (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),
                (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),
                255
            );
        }
    }
}

// ============================================================================
// SETUP SCENA - Definizione degli oggetti 3D
// ============================================================================
/*
 * Questa funzione crea la scena 3D:
 * - 7 sfere con diversi materiali e posizioni
 * - 2 luci per illuminazione realistica
 * 
 * I dati vengono copiati sia sulla GPU (per CUDA) che sulla CPU (per il
 * fallback CPU). Questa duplicazione è necessaria per supportare entrambe
 * le modalità di rendering.
 * 
 * MEMORIA CUDA:
 * =============
 * cudaMalloc: alloca memoria sulla GPU (device memory)
 * cudaMemcpy: copia dati tra CPU e GPU
 *   - cudaMemcpyHostToDevice: CPU → GPU
 *   - cudaMemcpyDeviceToHost: GPU → CPU (non usato qui)
 */
void setupScene() {
    Sphere spheres[NUM_SPHERES];
    
    /*
     * TRUCCO: Pavimento con sfera gigante!
     * ====================================
     * Invece di implementare un piano infinito, usiamo una sfera
     * con raggio enorme (1000 unità). La superficie è talmente
     * curva poco che appare piatta nella zona visibile.
     * 
     * Centro a y=-1000, raggio=1000 → superficie a y=0
     */
    spheres[0] = Sphere(Vec3(0, -1000, 0), 1000.0f, 
                        Material(Vec3(0.6f, 0.55f, 0.5f), DIFFUSE));
    
    /*
     * Sfera specchio centrale - materiale REFLECTIVE
     * Il valore 0.95 indica 95% di riflettività (specchio quasi perfetto)
     */
    spheres[1] = Sphere(Vec3(0, 1, 0), 1.0f, 
                        Material(Vec3(0.95f, 0.95f, 0.98f), REFLECTIVE, 0.95f));
    
    // Sfera rossa diffusa - superficie opaca che diffonde la luce
    spheres[2] = Sphere(Vec3(-2.5f, 0.7f, 1), 0.7f, 
                        Material(Vec3(0.95f, 0.2f, 0.2f), DIFFUSE));
    
    // Sfera verde semi-riflettente (40% riflessione, 60% diffusa)
    spheres[3] = Sphere(Vec3(2.5f, 0.8f, 0.5f), 0.8f, 
                        Material(Vec3(0.2f, 0.95f, 0.3f), REFLECTIVE, 0.4f));
    
    // Sfera blu diffusa
    spheres[4] = Sphere(Vec3(-1.0f, 0.5f, 2.5f), 0.5f, 
                        Material(Vec3(0.2f, 0.3f, 0.95f), DIFFUSE));
    
    // Sfera gialla riflettente (60% riflessione)
    spheres[5] = Sphere(Vec3(1.5f, 0.4f, 2.0f), 0.4f, 
                        Material(Vec3(1.0f, 0.9f, 0.15f), REFLECTIVE, 0.6f));
    
    // Sfera viola diffusa
    spheres[6] = Sphere(Vec3(0, 0.6f, -2.0f), 0.6f, 
                        Material(Vec3(0.8f, 0.25f, 0.9f), DIFFUSE));

    /*
     * LUCI
     * ====
     * Due luci per illuminazione a "tre punti" (senza la terza):
     * - Key light (luce principale): più intensa, simula il sole
     * - Fill light (luce di riempimento): più debole, ammorbidisce le ombre
     * 
     * Intensity > 1.0 per effetto HDR (High Dynamic Range)
     */
    Light lights[2];
    lights[0] = Light(Vec3(5, 10, 5),       // Posizione: alto a destra
                      Vec3(1.0f, 0.98f, 0.95f),  // Colore: bianco caldo (sole)
                      2.5f);                      // Intensità alta
    lights[1] = Light(Vec3(-8, 6, -3),      // Posizione: alto a sinistra-dietro
                      Vec3(0.7f, 0.8f, 1.0f),    // Colore: blu tenue (cielo)
                      1.2f);                      // Intensità media

    /*
     * ALLOCAZIONE E COPIA MEMORIA GPU
     * ================================
     * cudaMalloc alloca memoria nella VRAM della GPU.
     * cudaMemcpy copia i dati dalla RAM alla VRAM.
     * 
     * IMPORTANTE: La memoria GPU è separata dalla memoria CPU!
     * Non puoi accedere a d_spheres dalla CPU né a h_spheres dalla GPU.
     */
    cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Sphere));
    cudaMalloc(&d_lights, 2 * sizeof(Light));
    cudaMemcpy(d_spheres, spheres, NUM_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights, 2 * sizeof(Light), cudaMemcpyHostToDevice);

    // Copia dati anche su CPU per la modalità CPU rendering
    h_spheres = new Sphere[NUM_SPHERES];
    h_lights = new Light[2];
    h_output = new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT];  // Buffer immagine CPU
    memcpy(h_spheres, spheres, NUM_SPHERES * sizeof(Sphere));
    memcpy(h_lights, lights, 2 * sizeof(Light));
    
    // ========================================================================
    // CARICAMENTO MODELLO 3D DA FILE OBJ
    // ========================================================================
    /*
     * Carichiamo il diorama di Star Wars dal file OBJ.
     * Materiale: plastica opaca con leggera lucentezza
     * - Colore bianco/grigio tipico Stormtrooper
     * - Riflettività bassa (10%) per un look plastica opaca naturale
     */
    ObjMesh mesh;
    Material meshMaterial(Vec3(0.88f, 0.88f, 0.90f), REFLECTIVE, 0.10f);  // Plastica opaca con hint lucido
    
    // Prova a caricare il file OBJ
    if (mesh.load("StarWarsDiorama.obj", meshMaterial)) {
        // Trasforma il modello per posizionarlo nella scena
        // Centro a (0, 2, 0) con scala di 4 unità (adatta alla scena)
        mesh.transformToFit(Vec3(0, 2.0f, -1.0f), 4.0f);
        
        numTriangles = (int)mesh.triangles.size();
        
        // Salva i bounds per il test AABB (ottimizzazione!)
        meshBoundsMin = mesh.minBound;
        meshBoundsMax = mesh.maxBound;
        
        printf("\n=== MODELLO 3D CARICATO ===\n");
        printf("Triangoli totali: %d\n", numTriangles);
        printf("Bounding Box: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n",
               meshBoundsMin.x, meshBoundsMin.y, meshBoundsMin.z,
               meshBoundsMax.x, meshBoundsMax.y, meshBoundsMax.z);
        
        // Alloca e copia triangoli su GPU
        cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
        cudaMemcpy(d_triangles, mesh.triangles.data(), 
                   numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        
        // Copia triangoli su CPU per modalità CPU rendering
        h_triangles = new Triangle[numTriangles];
        memcpy(h_triangles, mesh.triangles.data(), numTriangles * sizeof(Triangle));
        
        // =====================================================================
        // COSTRUZIONE BVH (Bounding Volume Hierarchy)
        // =====================================================================
        /*
         * Il BVH è una struttura dati spaziale che accelera le intersezioni:
         * - Divide lo spazio in una griglia 3D (8x8x8 = 512 celle)
         * - Ogni cella contiene gli indici dei triangoli che la attraversano
         * - Un raggio attraversa solo le celle nel suo percorso (3D-DDA)
         * - Invece di testare ~7000 triangoli, ne testa ~100-800 per raggio
         * 
         * Risultato: speedup 5-10x rispetto a brute-force!
         */
        printf("Costruzione BVH (griglia %dx%dx%d)...\n", 
               BVH_GRID_SIZE, BVH_GRID_SIZE, BVH_GRID_SIZE);
        
        BVHBuilder builder;
        builder.build(mesh.triangles.data(), numTriangles, 
                      meshBoundsMin, meshBoundsMax);
        
        // Copia la griglia BVH (grid è membro pubblico)
        h_bvh = builder.grid;
        
        // Alloca e copia gli indici dei triangoli su GPU
        int totalIndices = (int)builder.allIndices.size();
        h_bvhIndices = new int[totalIndices];
        memcpy(h_bvhIndices, builder.allIndices.data(), totalIndices * sizeof(int));
        
        cudaMalloc(&d_bvhIndices, totalIndices * sizeof(int));
        cudaMemcpy(d_bvhIndices, h_bvhIndices, 
                   totalIndices * sizeof(int), cudaMemcpyHostToDevice);
        
        // Il BVH sulla GPU punta agli indici in memoria GPU
        h_bvh.triangleIndices = d_bvhIndices;
        
        printf("===========================\n\n");
    } else {
        printf("\n!!! ATTENZIONE: File OBJ non trovato !!!\n");
        printf("Assicurati che 'StarWarsDiorama.obj' sia nella cartella del progetto.\n");
        printf("Il raytracer funzionera' solo con le sfere.\n\n");
        numTriangles = 0;
        d_triangles = nullptr;
        h_triangles = nullptr;
        meshBoundsMin = Vec3(0, 0, 0);
        meshBoundsMax = Vec3(0, 0, 0);
        
        // Inizializza BVH vuoto
        h_bvh = BVHGrid();
        h_bvhIndices = nullptr;
        d_bvhIndices = nullptr;
    }
}

// ============================================================================
// CALLBACKS GLFW - Gestione input event-driven
// ============================================================================
/*
 * GLFW usa un sistema "event-driven" per l'input:
 * - Registri una funzione callback con glfwSetXxxCallback()
 * - GLFW chiama la tua funzione quando si verifica l'evento
 * 
 * Parametri comuni:
 * - win: la finestra che ha generato l'evento
 * - action: GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT
 * - mods: modificatori (Shift, Ctrl, Alt)
 */

/**
 * Callback tastiera - chiamata ad ogni pressione/rilascio tasto
 */
void keyCallback(GLFWwindow* win, int key, int scancode, int action, int mods) {
    // ESC: chiudi applicazione
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(win, GLFW_TRUE);  // Segnala al main loop di uscire
    }
    
    // R: reset camera alla posizione iniziale
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        camX = 0; camY = 2; camZ = 8;
        camYaw = 0; camPitch = 0;
        printf("Camera reset!\n");
    }
    
    // TAB: toggle tra rendering CPU e GPU
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        useCPU = !useCPU;
        printf("\n>>> Modalita' rendering: %s <<<\n\n", 
               useCPU ? "CPU (lento)" : "CUDA GPU (veloce)");
    }
    
    // F1: toggle pannello debug
    if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
        showDebugInfo = !showDebugInfo;
        printf("Debug info: %s\n", showDebugInfo ? "ON" : "OFF");
    }
    
    // F2: toggle modello 3D (mostra/nasconde lo stormtrooper)
    if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
        showMesh = !showMesh;
        printf("Modello 3D: %s\n", showMesh ? "ON" : "OFF");
    }
    
    // F3: toggle riflessioni (raytracing completo vs solo illuminazione diretta)
    if (key == GLFW_KEY_F3 && action == GLFW_PRESS) {
        enableReflections = !enableReflections;
        printf("Riflessioni: %s\n", enableReflections ? "ON (raytracing)" : "OFF (solo direct)");
    }
}

/**
 * Callback pulsanti mouse - gestisce cattura/rilascio mouse
 * 
 * "Mouse capture" = il cursore diventa invisibile e può muoversi
 * all'infinito (utile per controllare la camera in stile FPS).
 */
void mouseButtonCallback(GLFWwindow* win, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mouseCaptured = !mouseCaptured;  // Toggle stato
        
        if (mouseCaptured) {
            // Nascondi e cattura il cursore
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            firstMouse = true;  // Evita "salto" alla prossima lettura
            printf("Mouse catturato - muovi per guardare\n");
        } else {
            // Rilascia e mostra il cursore
            glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            printf("Mouse rilasciato\n");
        }
    }
}

/**
 * Callback movimento mouse - aggiorna rotazione camera
 * 
 * Questa funzione viene chiamata ogni volta che il mouse si muove.
 * Calcoliamo il "delta" (differenza) rispetto alla posizione precedente
 * e lo usiamo per ruotare la camera.
 */
void cursorPosCallback(GLFWwindow* win, double xpos, double ypos) {
    // Ignora se il mouse non è catturato
    if (!mouseCaptured) return;

    // Prima chiamata dopo la cattura? Inizializza senza ruotare.
    // Questo evita un "salto" dovuto alla differenza tra la posizione
    // precedente (centro schermo) e la posizione attuale.
    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
        return;
    }

    // Calcola quanto si è mosso il mouse
    float dx = (float)(xpos - lastMouseX);  // Positivo = destra
    float dy = (float)(ypos - lastMouseY);  // Positivo = giù
    lastMouseX = xpos;
    lastMouseY = ypos;

    // Aggiorna angoli camera
    camYaw += dx * mouseSpeed;    // Rotazione orizzontale
    camPitch -= dy * mouseSpeed;  // Rotazione verticale (invertita perché Y cresce verso il basso)

    // Limita pitch per evitare "gimbal lock" (capovolgimento camera)
    if (camPitch > 89.0f) camPitch = 89.0f;
    if (camPitch < -89.0f) camPitch = -89.0f;
}

// ============================================================================
// INPUT POLLING - Movimento camera
// ============================================================================
/*
 * Differenza tra POLLING e CALLBACK:
 * 
 * CALLBACK (usato per singoli eventi):
 * - "Dimmi quando premi ESC"
 * - Perfetto per azioni discrete (toggle, menu)
 * 
 * POLLING (usato qui per movimento continuo):
 * - "Il tasto W è premuto adesso?"
 * - Perfetto per movimento fluido (controllato ogni frame)
 * 
 * Per WASD usiamo polling perché vogliamo muoverci finché il tasto è tenuto.
 */
void processInput() {
    // Velocità = unità/secondo * deltaTime = movimento costante indipendente dal FPS!
    // Ctrl = movimento veloce (3x)
    float speed = moveSpeed * deltaTime * (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) ? 3.0f : 1.0f);
    
    /*
     * Calcola i vettori di movimento basati sulla direzione della camera.
     * Usiamo solo yaw (rotazione orizzontale) per il movimento,
     * così guardare in alto/basso non cambia la direzione di movimento.
     * 
     * forward = direzione "avanti" sul piano XZ
     * right = direzione "destra" sul piano XZ
     */
    float yawRad = camYaw * 3.14159f / 180.0f;  // Gradi → radianti
    Vec3 forward(sinf(yawRad), 0, -cosf(yawRad));  // Avanti nel piano XZ
    Vec3 right(cosf(yawRad), 0, sinf(yawRad));     // Destra nel piano XZ

    // WASD: movimento orizzontale
    if (glfwGetKey(window, GLFW_KEY_W)) { camX += forward.x * speed; camZ += forward.z * speed; }
    if (glfwGetKey(window, GLFW_KEY_S)) { camX -= forward.x * speed; camZ -= forward.z * speed; }
    if (glfwGetKey(window, GLFW_KEY_A)) { camX -= right.x * speed; camZ -= right.z * speed; }
    if (glfwGetKey(window, GLFW_KEY_D)) { camX += right.x * speed; camZ += right.z * speed; }
    
    // Space/Shift: movimento verticale
    if (glfwGetKey(window, GLFW_KEY_SPACE)) { camY += speed; }      // Sali
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) { camY -= speed; } // Scendi
}

// ============================================================================
// RENDER FRAME - Cuore del rendering loop
// ============================================================================
/*
 * Questa funzione viene chiamata ogni frame e:
 * 1. Calcola la direzione della camera dagli angoli yaw/pitch
 * 2. Renderizza la scena (GPU o CPU)
 * 3. Mostra il risultato sullo schermo tramite OpenGL
 * 
 * Il sistema CUDA-OpenGL interop evita copie di memoria:
 * - CUDA scrive direttamente nel PBO (memoria GPU)
 * - OpenGL legge dal PBO per aggiornare la texture
 * - Zero copie CPU coinvolte!
 */
void renderFrame() {
    /*
     * CALCOLO DIREZIONE CAMERA (da angoli Euler)
     * ==========================================
     * Yaw = rotazione attorno all'asse Y (sinistra/destra)
     * Pitch = rotazione attorno all'asse X (su/giù)
     * 
     * La formula converte angoli sferici in coordinate cartesiane:
     * - x = cos(pitch) * sin(yaw)  ← componente laterale
     * - y = sin(pitch)             ← componente verticale
     * - z = -cos(pitch) * cos(yaw) ← componente avanti/indietro (negativo perché -Z = avanti)
     */
    float yawRad = camYaw * 3.14159f / 180.0f;
    float pitchRad = camPitch * 3.14159f / 180.0f;

    Vec3 camDir(
        cosf(pitchRad) * sinf(yawRad),
        sinf(pitchRad),
        -cosf(pitchRad) * cosf(yawRad)
    );
    camDir = camDir.normalize();

    Vec3 camPos(camX, camY, camZ);
    Vec3 camUp(0, 1, 0);  // Vettore "su" del mondo (Y positivo)

    if (useCPU) {
        /*
         * MODALITÀ CPU (lenta)
         * =====================
         * Renderizza su buffer CPU, poi carica nella texture OpenGL.
         * Richiede copia CPU → GPU ogni frame.
         * NOTA: Con triangoli sarà MOLTO lento (~0.1 FPS)!
         */
        int rayDepth = enableReflections ? MAX_DEPTH : 1;
        if (numTriangles > 0 && showMesh) {
            renderCPUWithTriangles(h_output, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                   camPos, camDir, camUp, 
                                   h_spheres, NUM_SPHERES,
                                   h_triangles, numTriangles,
                                   h_lights, 2,
                                   meshBoundsMin, meshBoundsMax,
                                   rayDepth);
        } else {
            renderCPU(h_output, IMAGE_WIDTH, IMAGE_HEIGHT, camPos, camDir, camUp, h_spheres, h_lights, rayDepth);
        }
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, h_output);
    } else {
        /*
         * MODALITÀ CUDA GPU (veloce) - CUDA-OpenGL Interop
         * =================================================
         * 
         * 1. cudaGraphicsMapResources: "blocca" il PBO per uso CUDA
         *    (OpenGL non può accedervi durante questo tempo)
         * 
         * 2. cudaGraphicsResourceGetMappedPointer: ottieni puntatore GPU al PBO
         *    Ora CUDA può scrivere direttamente nella memoria del PBO!
         * 
         * 3. Lancia il kernel CUDA che scrive i pixel nel PBO
         * 
         * 4. cudaGraphicsUnmapResources: "sblocca" il PBO
         *    Ora OpenGL può usarlo per aggiornare la texture
         */
        uchar4* d_output;
        size_t numBytes;
        
        // Mappa il PBO per accesso CUDA
        cudaGraphicsMapResources(1, &cudaPBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_output, &numBytes, cudaPBO);

        /*
         * LANCIO KERNEL CUDA
         * ==================
         * dim3 block(16, 16): ogni blocco ha 16x16 = 256 thread
         * dim3 grid(...): calcola quanti blocchi servono per coprire l'immagine
         * 
         * Sintassi CUDA: kernel<<<grid, block>>>(argomenti...)
         * 
         * Con 1280x720:
         * - grid = (80, 45) blocchi
         * - totale = 80*45*256 = 921,600 thread!
         */
        // Blocchi più grandi per migliore occupancy GPU
        dim3 block(32, 8);  // 256 thread, layout ottimizzato per coalescenza memoria
        dim3 grid((IMAGE_WIDTH + block.x - 1) / block.x,
                  (IMAGE_HEIGHT + block.y - 1) / block.y);
        
        // Usa il kernel con BVH se abbiamo caricato un modello 3D
        if (numTriangles > 0) {
            /*
             * KERNEL CON BVH ACCELERATO
             * ==========================
             * Invece di testare tutti i triangoli per ogni raggio,
             * usiamo una griglia spaziale (BVH) per testare solo
             * i triangoli nelle celle attraversate dal raggio.
             * 
             * rayDepth: 1 = solo illuminazione diretta, MAX_DEPTH = con riflessioni
             */
            int rayDepth = enableReflections ? MAX_DEPTH : 1;
            renderKernelWithBVH<<<grid, block>>>(
                d_output, IMAGE_WIDTH, IMAGE_HEIGHT, 
                camPos, camDir, camUp, 
                d_spheres, NUM_SPHERES,
                d_triangles, numTriangles,
                d_lights, 2,
                h_bvh,      // BVH grid passato by value (contiene puntatore GPU)
                showMesh,
                rayDepth);  // Profondità raggi
        } else {
            // Fallback: kernel originale con sole sfere
            renderKernel<<<grid, block>>>(d_output, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                           camPos, camDir, camUp, d_spheres, d_lights);
        }
        
        // Aspetta che il kernel finisca (sincrono)
        cudaDeviceSynchronize();

        // Smappa il PBO - ora OpenGL può usarlo
        cudaGraphicsUnmapResources(1, &cudaPBO, 0);

        /*
         * AGGIORNAMENTO TEXTURE DA PBO
         * =============================
         * Bind del PBO come sorgente per glTexSubImage2D.
         * L'ultimo parametro "0" significa "leggi dal PBO invece che da un puntatore CPU".
         * 
         * Questo trasferimento avviene interamente sulla GPU!
         */
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    /*
     * DISEGNO TEXTURE A SCHERMO
     * =========================
     * Disegniamo un quadrato che copre tutto lo schermo (-1,-1) a (1,1)
     * con la texture mappata sopra.
     * 
     * Le coordinate texture (0,1)-(1,0) sono invertite verticalmente
     * perché OpenGL ha l'origine in basso a sinistra.
     */
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);  // Colore bianco (nessuna tinta)
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBegin(GL_QUADS);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);  // Basso-sinistra
        glTexCoord2f(1, 1); glVertex2f(1, -1);   // Basso-destra
        glTexCoord2f(1, 0); glVertex2f(1, 1);    // Alto-destra
        glTexCoord2f(0, 0); glVertex2f(-1, 1);   // Alto-sinistra
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// font5x7, drawChar, drawString sono definiti in debug_overlay.cuh

// ============================================================================
// DEBUG OVERLAY - Pannello informazioni a schermo
// ============================================================================
/*
 * Questa funzione disegna un pannello di debug sopra la scena 3D.
 * 
 * TECNICA: Rendering 2D sopra 3D
 * ==============================
 * 1. Salviamo lo stato delle matrici OpenGL (Push)
 * 2. Impostiamo una proiezione ortografica 2D
 * 3. Disegniamo il pannello e il testo
 * 4. Ripristiniamo lo stato precedente (Pop)
 * 
 * Proiezione ortografica vs prospettica:
 * - Prospettica: oggetti lontani appaiono più piccoli (3D realistico)
 * - Ortografica: nessuna prospettiva, perfetta per UI 2D
 */
void renderDebugOverlay() {
    // Non mostrare se disabilitato
    if (!showDebugInfo) return;

    /*
     * SETUP PROIEZIONE 2D
     * ====================
     * Salviamo le matrici correnti (prospettiva 3D) e impostiamo
     * una proiezione ortografica con coordinate pixel.
     * 
     * glOrtho(0, width, height, 0, ...): 
     * - Origine (0,0) in alto a sinistra (come le coordinate schermo)
     * - X cresce verso destra, Y cresce verso il basso
     */
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();  // Salva matrice proiezione corrente
    glLoadIdentity();
    glOrtho(0, IMAGE_WIDTH, IMAGE_HEIGHT, 0, -1, 1);  // Coordinate pixel
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();  // Salva matrice modelview corrente
    glLoadIdentity();

    // Setup per disegno 2D con trasparenza
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Trasparenza standard

    // Parametri layout pannello
    float scale = 2.0f;            // Scala del font
    float lineHeight = 10 * scale; // Altezza riga
    float panelWidth = 320;        // Larghezza pannello
    float panelHeight = 22 * lineHeight + 20;  // Altezza pannello (aumentata per triangoli)
    float startX = 20;             // Margine sinistro testo
    float startY = 25;             // Margine superiore testo

    // Sfondo pannello
    glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
    glBegin(GL_QUADS);
        glVertex2f(10, 10);
        glVertex2f(10 + panelWidth, 10);
        glVertex2f(10 + panelWidth, 10 + panelHeight);
        glVertex2f(10, 10 + panelHeight);
    glEnd();

    // Bordo
    glColor4f(0.0f, 1.0f, 0.5f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2f(10, 10);
        glVertex2f(10 + panelWidth, 10);
        glVertex2f(10 + panelWidth, 10 + panelHeight);
        glVertex2f(10, 10 + panelHeight);
    glEnd();

    char buf[128];
    float y = startY;

    // Titolo
    glColor3f(0.0f, 1.0f, 0.5f);
    drawString(startX, y, "DEBUG INFO [F1]", scale);
    y += lineHeight * 1.5f;

    // Modalità
    if (useCPU) {
        glColor3f(1.0f, 0.5f, 0.0f);
        drawString(startX, y, "MODE: CPU", scale);
    } else {
        glColor3f(0.0f, 1.0f, 0.3f);
        drawString(startX, y, "MODE: CUDA GPU", scale);
    }
    y += lineHeight;

    // FPS
    if (currentFPS >= 30) glColor3f(0.0f, 1.0f, 0.3f);
    else if (currentFPS >= 10) glColor3f(1.0f, 1.0f, 0.0f);
    else glColor3f(1.0f, 0.3f, 0.3f);
    sprintf(buf, "FPS: %d", currentFPS);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    glColor3f(0.9f, 0.9f, 0.9f);
    sprintf(buf, "Res: %dx%d", IMAGE_WIDTH, IMAGE_HEIGHT);
    drawString(startX, y, buf, scale);
    y += lineHeight * 1.5f;

    // GPU Info
    glColor3f(0.0f, 1.0f, 0.5f);
    drawString(startX, y, "=== GPU ===", scale);
    y += lineHeight;

    glColor3f(1.0f, 1.0f, 1.0f);
    sprintf(buf, "%s", gpuInfo.name);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    glColor3f(0.8f, 0.8f, 0.8f);
    sprintf(buf, "SM: %d  CC: %d.%d", gpuInfo.multiProcessorCount, 
            gpuInfo.computeCapabilityMajor, gpuInfo.computeCapabilityMinor);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    sprintf(buf, "VRAM: %.0f MB", gpuInfo.totalGlobalMem / (1024.0f * 1024.0f));
    drawString(startX, y, buf, scale);
    y += lineHeight;

    sprintf(buf, "Clock: %d MHz", gpuInfo.clockRateMHz);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    sprintf(buf, "Threads/Blk: %d", gpuInfo.maxThreadsPerBlock);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    sprintf(buf, "Warp: %d", gpuInfo.warpSize);
    drawString(startX, y, buf, scale);
    y += lineHeight * 1.5f;

    // Scene Info
    glColor3f(0.0f, 1.0f, 0.5f);
    drawString(startX, y, "=== SCENE ===", scale);
    y += lineHeight;

    glColor3f(1.0f, 1.0f, 0.0f);  // Giallo per sfere (sempre visibili)
    sprintf(buf, "Spheres: %d", NUM_SPHERES);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    if (numTriangles > 0) {
        if (showMesh) {
            glColor3f(0.0f, 1.0f, 1.0f);  // Ciano = attivo
            sprintf(buf, "Mesh: %d tris [F2=off]", numTriangles);
        } else {
            glColor3f(0.5f, 0.5f, 0.5f);  // Grigio = disattivato
            sprintf(buf, "Mesh: OFF [F2=on]");
        }
    } else {
        glColor3f(0.5f, 0.5f, 0.5f);
        sprintf(buf, "Mesh: 0 (no OBJ)");
    }
    drawString(startX, y, buf, scale);
    y += lineHeight * 1.5f;

    // Rendering Options
    glColor3f(0.0f, 1.0f, 0.5f);
    drawString(startX, y, "=== OPTIONS ===", scale);
    y += lineHeight;

    if (enableReflections) {
        glColor3f(0.0f, 1.0f, 0.0f);  // Verde = attivo
        sprintf(buf, "Reflections: ON [F3]");
    } else {
        glColor3f(0.5f, 0.5f, 0.5f);  // Grigio = disattivato
        sprintf(buf, "Reflections: OFF [F3]");
    }
    drawString(startX, y, buf, scale);
    y += lineHeight * 1.5f;

    // Camera Info
    glColor3f(0.0f, 1.0f, 0.5f);
    drawString(startX, y, "=== CAMERA ===", scale);
    y += lineHeight;

    glColor3f(0.8f, 0.8f, 0.8f);
    sprintf(buf, "Pos: %.1f %.1f %.1f", camX, camY, camZ);
    drawString(startX, y, buf, scale);
    y += lineHeight;

    sprintf(buf, "Yaw: %.0f Pitch: %.0f", camYaw, camPitch);
    drawString(startX, y, buf, scale);

    // Cleanup: disabilita blend e ripristina stato OpenGL
    glDisable(GL_BLEND);

    // Ripristina le matrici salvate (torna alla proiezione 3D)
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();  // Ripristina proiezione 3D
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();  // Ripristina modelview
}

// ============================================================================
// INIT OPENGL - Setup PBO e Texture
// ============================================================================
/*
 * Questa funzione inizializza le risorse OpenGL necessarie per
 * visualizzare l'output del ray tracer:
 * 
 * 1. PBO (Pixel Buffer Object): buffer GPU condiviso con CUDA
 * 2. Texture: immagine visualizzata sullo schermo
 * 3. Registrazione CUDA-OpenGL: "connette" il PBO a CUDA
 */
bool initGL() {
    /*
     * CREAZIONE PBO (Pixel Buffer Object)
     * ====================================
     * Un PBO è un buffer nella memoria GPU che può essere usato come
     * sorgente/destinazione per operazioni sui pixel.
     * 
     * GL_PIXEL_UNPACK_BUFFER: usato come sorgente per caricare texture
     * GL_DYNAMIC_DRAW: i dati cambieranno spesso (ogni frame)
     */
    glGenBuffers(1, &pbo);  // Genera un ID per il buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);  // Rendi attivo
    glBufferData(GL_PIXEL_UNPACK_BUFFER,        // Tipo buffer
                 IMAGE_WIDTH * IMAGE_HEIGHT * 4, // Dimensione (RGBA = 4 byte/pixel)
                 nullptr,                        // Nessun dato iniziale
                 GL_DYNAMIC_DRAW);               // Pattern di utilizzo
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);    // Sbind

    /*
     * REGISTRAZIONE CUDA-OPENGL
     * =========================
     * Questa chiamata "registra" il PBO con CUDA, permettendo a CUDA
     * di accedere alla stessa memoria del PBO senza copie.
     * 
     * cudaGraphicsMapFlagsWriteDiscard: CUDA scriverà nel buffer
     * (CUDA non ha bisogno di leggere il contenuto precedente)
     */
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    /*
     * CREAZIONE TEXTURE
     * =================
     * La texture conterrà l'immagine renderizzata e verrà disegnata
     * su un quadrato a schermo intero.
     * 
     * GL_LINEAR: interpolazione bilineare quando si ridimensiona
     * GL_RGBA8: 4 canali (Red, Green, Blue, Alpha), 8 bit ciascuno
     */
    glGenTextures(1, &textureID);  // Genera ID texture
    glBindTexture(GL_TEXTURE_2D, textureID);  // Rendi attiva
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // Filtro riduzione
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // Filtro ingrandimento
    glTexImage2D(GL_TEXTURE_2D,    // Tipo texture
                 0,                 // Livello mipmap (0 = base)
                 GL_RGBA8,          // Formato interno (GPU)
                 IMAGE_WIDTH, IMAGE_HEIGHT,  // Dimensioni
                 0,                 // Border (sempre 0)
                 GL_RGBA,           // Formato pixel sorgente
                 GL_UNSIGNED_BYTE,  // Tipo dati sorgente
                 nullptr);          // Nessun dato iniziale

    return true;
}

// ============================================================================
// CLEANUP - Deallocazione risorse
// ============================================================================
/*
 * IMPORTANTE: Liberare le risorse nell'ordine corretto!
 * 
 * 1. Prima de-registra le risorse CUDA-OpenGL
 * 2. Poi elimina le risorse OpenGL
 * 3. Poi libera la memoria CUDA
 * 4. Poi libera la memoria CPU
 * 5. Infine resetta il device CUDA
 * 
 * L'ordine è importante perché alcune risorse dipendono da altre.
 * cudaDeviceReset() deve essere l'ultima operazione CUDA.
 */
void cleanup() {
    // 1. De-registra interop CUDA-OpenGL
    if (cudaPBO) cudaGraphicsUnregisterResource(cudaPBO);
    
    // 2. Elimina risorse OpenGL
    if (pbo) glDeleteBuffers(1, &pbo);
    if (textureID) glDeleteTextures(1, &textureID);
    
    // 3. Libera memoria GPU (CUDA)
    if (d_spheres) cudaFree(d_spheres);
    if (d_lights) cudaFree(d_lights);
    if (d_triangles) cudaFree(d_triangles);
    if (d_bvhIndices) cudaFree(d_bvhIndices);  // BVH triangle indices
    
    // 4. Libera memoria CPU
    if (h_spheres) delete[] h_spheres;
    if (h_lights) delete[] h_lights;
    if (h_output) delete[] h_output;
    if (h_triangles) delete[] h_triangles;
    if (h_bvhIndices) delete[] h_bvhIndices;  // BVH triangle indices CPU
    
    // 5. Reset completo del device CUDA
    // (libera tutte le risorse CUDA rimanenti)
    cudaDeviceReset();
}

// ============================================================================
// MAIN - Entry point dell'applicazione
// ============================================================================
/*
 * CICLO DI VITA DELL'APPLICAZIONE:
 * =================================
 * 
 * 1. INIZIALIZZAZIONE
 *    - GLFW (finestra e input)
 *    - GLEW (funzioni OpenGL moderne)
 *    - CUDA (device GPU)
 *    - OpenGL (PBO e texture)
 *    - Scena (sfere e luci)
 * 
 * 2. MAIN LOOP (ripetuto finché la finestra è aperta)
 *    - Calcola deltaTime
 *    - Processa input
 *    - Renderizza frame
 *    - Disegna debug overlay
 *    - Scambia buffer (double buffering)
 *    - Controlla eventi
 * 
 * 3. CLEANUP
 *    - Libera risorse in ordine inverso
 */
int main() {
    printf("===========================================\n");
    printf("   CUDA Ray Tracer - Real-time Interattivo\n");
    printf("===========================================\n\n");

    /*
     * INIZIALIZZAZIONE GLFW
     * =====================
     * GLFW deve essere inizializzato prima di qualsiasi altra chiamata GLFW.
     * Restituisce false se qualcosa va storto (es. driver video mancante).
     */
    if (!glfwInit()) {
        printf("Errore: GLFW init fallito!\n");
        return -1;
    }

    /*
     * CREAZIONE FINESTRA
     * ==================
     * glfwCreateWindow crea una finestra E un contesto OpenGL.
     * I parametri nullptr alla fine sono per:
     * - Monitor (nullptr = finestra normale, non fullscreen)
     * - Condivisione contesto (nullptr = nessuna condivisione)
     */
    window = glfwCreateWindow(IMAGE_WIDTH, IMAGE_HEIGHT, 
        "CUDA Ray Tracer [Click sinistro: cattura mouse | ESC: esci]", nullptr, nullptr);
    if (!window) {
        printf("Errore: creazione finestra fallita!\n");
        glfwTerminate();
        return -1;
    }

    /*
     * CONTESTO OPENGL
     * ===============
     * Un "contesto OpenGL" è lo stato interno di OpenGL per una finestra.
     * Deve essere reso "corrente" prima di usare qualsiasi funzione OpenGL.
     * 
     * glfwSwapInterval(0): disabilita V-Sync per massimi FPS
     * (con V-Sync attivo saresti limitato a 60 FPS sul monitor)
     */
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // 0 = no V-Sync, 1 = V-Sync attivo

    /*
     * INIZIALIZZAZIONE GLEW
     * =====================
     * GLEW carica le funzioni OpenGL "estese" (versioni > 1.1).
     * Senza GLEW, funzioni come glGenBuffers non sarebbero disponibili
     * perché non sono nella libreria OpenGL di sistema.
     * 
     * NOTA: GLEW deve essere inizializzato DOPO aver creato il contesto OpenGL!
     */
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        printf("Errore GLEW: %s\n", glewGetErrorString(glewErr));
        return -1;
    }

    /*
     * INIZIALIZZAZIONE CUDA
     * =====================
     * cudaSetDevice seleziona quale GPU usare (0 = prima GPU).
     * Necessario prima di qualsiasi operazione CUDA.
     * 
     * In un sistema multi-GPU potresti scegliere quale usare:
     * - GPU 0: integrata (Intel)
     * - GPU 1: dedicata (NVIDIA)
     */
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice fallito: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }

    /*
     * QUERY PROPRIETÀ GPU
     * ====================
     * cudaGetDeviceProperties riempie una struttura con tutte le
     * informazioni sulla GPU: nome, memoria, capacità, limiti, ecc.
     * 
     * Queste info sono utili per:
     * - Debug e profiling
     * - Ottimizzazione (es. scegliere block size ottimale)
     * - Verificare compatibilità
     */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Copia info nella struttura globale per il debug overlay
    strncpy(gpuInfo.name, prop.name, sizeof(gpuInfo.name) - 1);
    gpuInfo.computeCapabilityMajor = prop.major;  // Es: 7 per Turing (RTX 20xx)
    gpuInfo.computeCapabilityMinor = prop.minor;  // Es: 5 per RTX 2050
    gpuInfo.multiProcessorCount = prop.multiProcessorCount;  // Numero di SM
    gpuInfo.totalGlobalMem = prop.totalGlobalMem;  // VRAM in bytes
    gpuInfo.maxThreadsPerBlock = prop.maxThreadsPerBlock;  // Max 1024 tipicamente
    gpuInfo.maxThreadsPerMP = prop.maxThreadsPerMultiProcessor;
    
    // La frequenza clock richiede una chiamata separata
    int clockRateKHz = 0;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
    gpuInfo.clockRateMHz = clockRateKHz / 1000;
    
    gpuInfo.sharedMemPerBlock = (int)prop.sharedMemPerBlock;
    gpuInfo.warpSize = prop.warpSize;  // 32 per tutte le GPU NVIDIA
    
    // Stampa info a console
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Risoluzione: %dx%d\n\n", IMAGE_WIDTH, IMAGE_HEIGHT);

    printf("CONTROLLI:\n");
    printf("  Click sinistro - Cattura/rilascia mouse\n");
    printf("  WASD           - Movimento\n");
    printf("  Spazio/Shift   - Su/Giu\n");
    printf("  Ctrl           - Movimento veloce\n");
    printf("  TAB            - Cambia CPU/CUDA\n");
    printf("  F1             - Toggle debug info\n");
    printf("  R              - Reset camera\n");
    printf("  ESC            - Esci\n\n");

    // Inizializza OpenGL (PBO + Texture)
    if (!initGL()) {
        printf("Errore inizializzazione OpenGL!\n");
        return -1;
    }

    // Crea la scena 3D (sfere + luci)
    setupScene();

    /*
     * REGISTRAZIONE CALLBACKS
     * =======================
     * Diciamo a GLFW quali funzioni chiamare per gestire gli eventi.
     * GLFW salva i puntatori alle funzioni e le chiama quando serve.
     */
    glfwSetKeyCallback(window, keyCallback);          // Eventi tastiera
    glfwSetMouseButtonCallback(window, mouseButtonCallback);  // Click mouse
    glfwSetCursorPosCallback(window, cursorPosCallback);      // Movimento mouse

    printf("Rendering avviato! Clicca nella finestra per iniziare.\n\n");

    /*
     * MAIN LOOP - Il cuore dell'applicazione
     * ======================================
     * Questo è il classico "game loop" usato in tutti i giochi e
     * applicazioni real-time:
     * 
     *   while (finestra aperta) {
     *       1. Calcola tempo passato
     *       2. Processa input
     *       3. Aggiorna logica (non usato qui)
     *       4. Renderizza
     *       5. Mostra risultato
     *       6. Controlla eventi
     *   }
     * 
     * DOUBLE BUFFERING:
     * Il rendering avviene su un buffer nascosto (back buffer).
     * glfwSwapBuffers scambia i buffer, mostrando l'immagine completa.
     * Questo evita lo "screen tearing" (immagine a metà).
     */
    while (!glfwWindowShouldClose(window)) {
        /*
         * CALCOLO DELTA TIME
         * ==================
         * deltaTime = tempo trascorso dall'ultimo frame.
         * Usato per movimento indipendente dal framerate.
         * 
         * Esempio:
         * - A 60 FPS: deltaTime ≈ 0.016s
         * - A 30 FPS: deltaTime ≈ 0.033s
         * - Movimento = velocità * deltaTime (costante in entrambi i casi)
         */
        float currentFrame = (float)glfwGetTime();  // Secondi dall'avvio
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        /*
         * CALCOLO FPS
         * ===========
         * Contiamo i frame per un secondo, poi aggiorniamo il display.
         * Questo dà un valore stabile invece di "FPS istantanei" rumorosi.
         */
        frameCount++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0f) {
            currentFPS = frameCount;  // Frame nell'ultimo secondo
            
            // Aggiorna titolo finestra con FPS e modalità
            char title[256];
            sprintf(title, "Ray Tracer [%s] | FPS: %d | F1: debug | TAB: mode", 
                    useCPU ? "CPU" : "CUDA GPU", currentFPS);
            glfwSetWindowTitle(window, title);
            
            // Reset contatori per il prossimo secondo
            frameCount = 0;
            fpsTimer = 0;
        }

        // Processa input da tastiera (WASD, etc.)
        processInput();
        
        // Renderizza la scena 3D (il lavoro pesante!)
        renderFrame();
        
        // Disegna il pannello debug sopra la scena
        renderDebugOverlay();

        /*
         * SWAP BUFFERS + POLL EVENTS
         * ==========================
         * glfwSwapBuffers: scambia front/back buffer (mostra l'immagine)
         * glfwPollEvents: processa eventi di sistema (resize, close, input)
         */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    /*
     * CLEANUP E TERMINAZIONE
     * ======================
     * Liberiamo tutte le risorse in ordine (quasi) inverso rispetto
     * all'allocazione. Questo è importante per evitare memory leak
     * e crash.
     */
    cleanup();  // Libera risorse CUDA e OpenGL
    glfwDestroyWindow(window);  // Distruggi la finestra
    glfwTerminate();  // Termina GLFW e libera le sue risorse

    printf("\nChiusura completata.\n");
    return 0;  // Successo!
}
