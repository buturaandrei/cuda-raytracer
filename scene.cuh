/**
 * scene.cuh - Gestione della Scena
 * 
 * Questo file contiene le definizioni per configurare la scena 3D:
 * - Sfere con materiali diversi
 * - Luci
 * - Parametri globali
 * 
 * LA SCENA DEL RAYTRACER:
 * =======================
 * 
 * La nostra scena contiene:
 * 
 * 1. TERRENO (ground plane):
 *    - Una sfera ENORME (raggio 1000) centrata molto sotto
 *    - L'effetto è una superficie piatta infinita
 * 
 * 2. SFERE:
 *    - Una sfera specchio centrale (riflette l'ambiente)
 *    - Sfere colorate con materiali diversi
 * 
 * 3. LUCI:
 *    - Luce principale (come il sole)
 *    - Luce di riempimento (fill light) - evita ombre troppo scure
 * 
 * LAYOUT DELLA SCENA (vista dall'alto):
 * =====================================
 * 
 *               ^ Z negativo
 *               |
 *     [Purple]  |
 *               |
 *  [Red]   [MIRROR]   [Green]    → X positivo
 *               |
 *     [Blue]   |  [Yellow]
 *               |
 *               
 *  La camera parte a Z=8 guardando verso Z negativo
 */

#ifndef SCENE_CUH
#define SCENE_CUH

#include "vec3.cuh"
#include "materials.cuh"
#include "sphere.cuh"
#include "light.cuh"

// ============================================================================
// PARAMETRI DELLA SCENA
// ============================================================================

// Dimensione dell'immagine renderizzata
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 1280
#endif

#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 720
#endif

// Profondità massima per le riflessioni
// Più alto = più rimbalzi = più realistico ma più lento
#ifndef MAX_DEPTH
#define MAX_DEPTH 10
#endif

// Numero di sfere nella scena
#ifndef NUM_SPHERES
#define NUM_SPHERES 7
#endif

/**
 * Crea e restituisce l'array delle sfere della scena
 * 
 * NOTA: Questa funzione è un esempio di come configurare la scena.
 * Nel codice reale, le sfere vengono create in setupScene().
 * 
 * MATERIALI USATI:
 * - DIFFUSE: superfici opache (rossa, blu, viola)
 * - REFLECTIVE: superfici specchiate (specchio centrale, verde, giallo)
 * 
 * TRUCCO DEL GROUND PLANE:
 * ========================
 * Invece di implementare un piano infinito (più complesso matematicamente),
 * usiamo una sfera ENORME (raggio 1000) centrata a Y = -1000.
 * 
 * Il punto più alto della sfera è a Y = -1000 + 1000 = 0
 * Quindi la "superficie" del terreno è a Y = 0.
 * 
 * La curvatura è talmente leggera che non si nota.
 */

// Configurazione delle sfere (valori di esempio)
// spheres[0] = Ground plane (sfera grande che funge da pavimento)
// spheres[1] = Specchio centrale
// spheres[2] = Sfera rossa
// spheres[3] = Sfera verde semi-riflettente
// spheres[4] = Sfera blu
// spheres[5] = Sfera gialla riflettente
// spheres[6] = Sfera viola

/**
 * LUCI NELLA SCENA:
 * =================
 * 
 * Luce 0 - "Sole":
 *   Posizione: in alto a destra e davanti
 *   Colore: leggermente caldo (bianco tendente al giallo)
 *   Intensità: 2.5 (alta)
 * 
 * Luce 1 - "Fill Light":
 *   Posizione: in alto a sinistra e dietro
 *   Colore: leggermente freddo (bianco tendente al blu)
 *   Intensità: 1.2 (media)
 * 
 * La combinazione di queste due luci crea:
 * - Ombre morbide (perché illuminate da più direzioni)
 * - Contrasto interessante (colori caldi vs freddi)
 * - Profondità visiva (la fill light rivela dettagli in ombra)
 */

#endif // SCENE_CUH
