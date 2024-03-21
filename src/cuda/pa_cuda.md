# PA2

- [1. Úvod](#1-úvod)
  - [1.1. Amdahlův zákon](#11-amdahlův-zákon)
  - [1.2. N-Body Problem](#12-n-body-problem)
  - [1.3. Boiler Problem](#13-boiler-problem)
  - [1.4. Dining Philosophers Problem](#14-dining-philosophers-problem)
  - [1.5. nvcc](#15-nvcc)
  - [1.6. VisualStudio22](#16-visualstudio22)
  - [1.7. Limity GPU](#17-limity-gpu)
  - [1.8. Jak vybrat N náhodných unikátních čísel z pole?](#18-jak-vybrat-n-náhodných-unikátních-čísel-z-pole)
- [2. Technologie CUDA](#2-technologie-cuda)
  - [2.1. Práce s vektory](#21-práce-s-vektory)
  - [2.2. Shared memory (SH)](#22-shared-memory-sh)
    - [2.2.1. Statická shared memory](#221-statická-shared-memory)
    - [2.2.2. Dynamická shared memory](#222-dynamická-shared-memory)
  - [2.3. Parallel Reduction](#23-parallel-reduction)
  - [2.4. Zarovnaná paměť](#24-zarovnaná-paměť)
  - [2.5. Bank Conflicts](#25-bank-conflicts)
  - [2.6. Constant Memory](#26-constant-memory)
- [3. Examples](#3-examples)

## 1. Úvod

Typy paralelizmu:

1. **Datový paralelismus**.
2. **Instrukční paralelismus** - využití instrukcí - např. jedny vlákna chystají data, další je zpracovávají.

**Logické vlákno** je *sled instrukcí*. Potřebuju **registry** a nějakou výpočetní jednotku. Běží dokud má instrukce. Přerušení výpočtu vláken určuje programátor.

**Pointer** je proměnná, jejíž hodnotou je adresa.

**Hyper-threading** - každé vlákno se navenek rozdělí. 16jádro má 16 instrukčních sad.

Co rozumíme pojmem **proces**? OS *alokuje a spravuje paměť*, přidělí *stack* a alespoň jeden *main thread*.

Bloky jsou schedulované pomocí Streaming Multiprocessoru (SM). GPU má pouze omezený počet SM *(NVIDIA GeForce GTX 1650 má 14 SM)*.

Skrýváním latence (čekání, **latency hiding**) rozumíme zkrácení nečinnosti procesoru. Instrukce mají nějaký čas vykonávání (např. odmocnina nebo modulo je drahá instrukce). Čtení z disku jakožto nejdražší paměťová operace.

32 CUDA vláken běží se stejnou instrukční sadou ve **warpu**.

### 1.1. Amdahlův zákon

Maximální teoretické zrychlení pomocí paralelismu:

$$\boxed{S=\dfrac{1}{r_s+\dfrac{r_p}{n}}}$$

- $S$ - *speed-up*
- $r_s$ - *serial runtime* (čas sekvenčního algoritmu)
- $r_p$ - *parallel runtime* (čas paralelního běhu)
- $n$ - *number of cores*

Příklad: 70 % programu běží sériově. Máme k dispozici 8 jader.

$$ S=\dfrac{1}{0.7+\dfrac{0.3}{8}} \approx 1.35 $$

### 1.2. N-Body Problem

Výpočet gravitačních interakcí těles, kdy musíme počítat interakce každý s každým (neexistuje matematický model pro $N$ těles).

### 1.3. Boiler Problem

Buď kotel na vodu a dvě kontrolní vlákna. Problém nelze řešit pouze těmito dvěmi vlákny. To, co chce udělat jedno z nich, chce i to druhé. Musí tam být nějaký další prvek, který bude vlákna ovládat (např. semafor, mutex).

### 1.4. Dining Philosophers Problem

<img src="figures/dining-philosophers.png" alt="dining-philosophers" width="200px">

### 1.5. nvcc

- Compiler pro rozšíření CUDA.
- Vezme zdroják a rozdělí kód na funkce, které se mají kompilovat pomocí `g++` a `cudapp`.

### 1.6. VisualStudio22

<img src="figures/vs22-setup.png" alt="vs22-setup" width="200px">

- RMB na projekt, `unload`, upravit verze CUDA, `reload`, `rebuild`.

<img src="figures/vs22-properties.png" alt="vs22-properties" width="400px">

- *Compute capability* je dána modelem GPU.

### 1.7. Limity GPU

```text
[GPU details]:
  Clock rate                                        : 1.51 GHz
  Number of multiprocessors                         : 14
  Number of cores                                   : 896
  Warp size                                         : 32
  Total amount of global memory                     : 4095 Mb
  Total amount of constant memory                   : 65536 bytes
  Maximum memory pitch                              : 2147483647 bytes
  Texture alignment                                 : 512 bytes
  Run time limit on kernels                         : Yes
  Integrated                                        : No
  Support host page-locked memory mapping           : Yes
  Compute mode                                      : Default (multiple host threads can use this device simultaneously)

[SM details]:
  Number of cores                                   : 64
  Total amount of shared memory per SM              : 65536 bytes
  Total number of registers available per SM        : 65536
  Maximum number of resident blocks per SM          : 16
  Maximum number of resident threads per SM         : 1024
  Maximum number of resident warps per SM           : 32

[BLOCK details]:
  Total amount of shared memory per block           : 49152 bytes
  Total number of registers available for block     : 65536
  Maximum number of threads per block               : 1024
  Maximum sizes of each dimension of a block        : 1024, 1024, 64
  Maximum sizes of each dimension of a grid         : 2147483647, 65535, 65535

SELECTED GPU Device 0: "NVIDIA GeForce GTX 1650" with compute capability 7.5
```

### 1.8. Jak vybrat N náhodných unikátních čísel z pole?

- Dvě pole: originální a pole náhodných hodnot. Sortuju náhodné pole a podle toho měním pozice v originálním poli. Potom vyberu prvních $N$ hodnot z takto vytvořeného pole.

## 2. Technologie CUDA

- Grid se rozpadne na bloky.
- Bloky se rozdělí na warpy. Jeden warp má 32 vláken, které jsou schopny vykonávat SIMT (Single Instruction Multiple Thread).
- `__syncthreads()` - čeká se až všechny warpy dokončí výpočet.

Jeden **streaming multiprocessor (SM) zpracovává jeden blok**. Při dělení problému řádu $N$ obvykle dádá smysl rozdělit úlohu na násobek $|SM|$ bloků, kde $|SM|$ je počet SM.

**Blok je dělený do warpů po 32 vláknech.** Vlákna v rámci jednoho warpu jsou **synchronní** (tzn. nedává smysl volat `__syncthreads()`).

**Global memory** je hodně, ale je pomalá.

**Karta čte obvykle po 512 B.**

### 2.1. Práce s vektory

- Zvolím grid, např. $(2,1,1)$.
- Zvolím velikost bloku např. $(128,1,1)$. Není důvod to komplikovat více dimenzemi.

```cpp
unsigned int tid = blockIdx.x * blockDim + threadIdx;
```

### 2.2. Shared memory (SH)

Shared memory (SH) je alokovaná pro každý thread block, všechny vlákna v rámci bloku mají přístup do stejné sdílené paměti. Latence SH je přibližně 10x nižší oproti globální paměti, která není načtená v cache (pokud nedochází k *bank konfliktům* mezi vlákny). Každý SM má k dispozici 64 KB shared memory.

Multicast přístup k SH znamená, že přístup do paměti na stejnou pozici více vlákny je v rámci warpu obsloužen současně.

- `__syncthreads()`
- `volatile` řekne compileru, že se nemá provádět cache hodnot, používá se při paralelní redukci

#### 2.2.1. Statická shared memory

V kernelu: `__shared__ int vec[256];`

#### 2.2.2. Dynamická shared memory

V kernelu: `extern __shared__ int x[];` (`extern` a prázdné `[]`)

Při volání kernelu specifikujeme třetí volitený parametr pro velikost SH pro každý blok (v B): `kernel<<<nBlocks, nThreadsPerBlock, nBytesSH>>>(...)`, např. `kernel<<<1, n, n*sizeof(int)>>>(...)`.

Jak postupovat, když potřebuju více SH polí v rámci kernelu?

```cpp
extern __shared__ int s[];
int *integerData = s;                        // nI ints
float *floatData = (float*)&integerData[nI]; // nF floats
char *charData = (char*)&floatData[nF];      // nC chars

myKernel<<<
  gridSize, 
  blockSize, 
  nI*sizeof(int)+nF*sizeof(float)+nC*sizeof(char)
>>>(...);
```

### 2.3. Parallel Reduction

Průchod dat - agregace do jedné hodnoty (min, max, suma atd.)

1. Vlákna zkopírují hodnoty z globalní paměti do shared memory.

Hledání minima:

<img src="figures/parallel-reduction.png" alt="parallel-reduction" width="400px">

### 2.4. Zarovnaná paměť

```cuda
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
```

Sloupcová vs. řádková matice - do 1D se matice ukládá buď po sloupcích *(column major)* nebo po řádcích *(row major)*.

### 2.5. Bank Conflicts

- Přístup k paměti přes nějaký bank. Lze optimalizovat pomocí vhodného indexování prvků.

### 2.6. Constant Memory

- Klíčové slovo `__constant__`, např. `__constant__ __device__ int myVar;`
- Chová se jako konstanta z pohledu GPU, ale lze měnit z HOST.
- `cudaMemCpyToSymbol` a `cudaMemCpyFromSymbol`
- Přístup k poli uloženém v konstantní paměti se serializuje! V tomto případě je lepší SH, která podporuje multicast.

## 3. Examples

<details><summary> Add vectors </summary>

```cpp
{{#include src/1_add_vectors.cu}}
```

</details>

<details><summary> Pitched memory </summary>

```cpp
{{#include src/2_malloc_pitch.cu}}
```

</details>

<details><summary> Parallel reduce: Raindrops </summary>

```cpp
{{#include src/3_raindrops.cu}}
```

</details>

<details><summary> Constant memory </summary>

```cpp
{{#include src/4_constant_memory.cu}}
```

</details>
