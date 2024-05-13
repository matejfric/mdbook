# PA2

- [1. Úvod](#1-úvod)
  - [1.1. Model vykonávání (Execution Model)](#11-model-vykonávání-execution-model)
  - [1.2. Amdahlův zákon](#12-amdahlův-zákon)
  - [1.3. N-Body Problem](#13-n-body-problem)
  - [1.4. Boiler Problem](#14-boiler-problem)
  - [1.5. Dining Philosophers Problem](#15-dining-philosophers-problem)
  - [1.6. nvcc](#16-nvcc)
  - [1.7. VisualStudio22](#17-visualstudio22)
  - [1.8. Jak vybrat N náhodných unikátních čísel z pole?](#18-jak-vybrat-n-náhodných-unikátních-čísel-z-pole)
- [2. Technologie CUDA](#2-technologie-cuda)
  - [2.1. Occupancy](#21-occupancy)
  - [2.2. Práce s vektory](#22-práce-s-vektory)
  - [2.3. Shared memory (SH)](#23-shared-memory-sh)
    - [2.3.1. Statická shared memory](#231-statická-shared-memory)
    - [2.3.2. Dynamická shared memory](#232-dynamická-shared-memory)
  - [2.4. Parallel Reduction](#24-parallel-reduction)
  - [2.5. Zarovnaná paměť](#25-zarovnaná-paměť)
  - [2.6. Bank Conflicts](#26-bank-conflicts)
  - [2.7. Constant Memory](#27-constant-memory)
- [3. Texturovací paměť](#3-texturovací-paměť)
  - [3.1. Normálové mapy (Normal Mapping)](#31-normálové-mapy-normal-mapping)
- [4. OpenGL](#4-opengl)
  - [4.1. Grafika](#41-grafika)
  - [4.2. Double Buffering](#42-double-buffering)
  - [4.3. GLUT library](#43-glut-library)
- [5. Atomické intrukce (Atomic Functions)](#5-atomické-intrukce-atomic-functions)
  - [5.1. CUDA atomické instrukce](#51-cuda-atomické-instrukce)
- [6. Page Lock Memory](#6-page-lock-memory)
- [7. Unified Memory](#7-unified-memory)
- [8. Streamy](#8-streamy)
  - [8.1. Memory Stream](#81-memory-stream)
  - [8.2. CUDA Streams](#82-cuda-streams)
- [9. CuBLAS](#9-cublas)
  - [9.1. N-body maticově](#91-n-body-maticově)
- [10. AoS vs. SoA](#10-aos-vs-soa)
- [11. Examples](#11-examples)

## 1. Úvod

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

> Proč potřebujeme více jader procesoru? **Skrývání latence** - každá instrukce má nějaký čas vykonávání a my chceme skrývat latenci mezi instrukcemi.

GPU (GP-GPU) poskytuje mnohem větší propustnost instrukcí a paměti oproti CPU podobné ceny.

Rozdíl ve schopnostech GPU a CPU existuje proto, že jsou navrženy s různými cíli. Zatímco CPU je navržen tak, aby vynikal co nejrychlejším prováděním sekvence operací, tzv. vláken, a může paralelně provádět několik desítek těchto vláken, GPU je navržen tak, aby vynikal paralelním prováděním tisíců těchto operací (úlohu rozdělí na velké množství jednotlivých vláken a menším výkonem a dosáhne tak vyšší propustnosti).

Na GPU je více tranzistorů věnováno zpracování dat namísto ukládání dat do mezipaměti *(cache)* a řízení toku *(if else)*.

<img src="figures/gpu-devotes-more-transistors-to-data-processing.png" alt="gpu-devotes-more-transistors-to-data-processing" width="500x">

Typy paralelizmu:

1. **Datový paralelismus** - data jsou rozdělena do bloků a každý blok je zpracován procesem/vláknem.
2. **Instrukční paralelismus** - využití nezávislých instrukcí - např. jedny vlákna chystají data, další je zpracovávají.

**Logické vlákno** je *sled instrukcí*. Potřebuju **registry** a nějakou výpočetní jednotku. Běží dokud má instrukce. Přerušení výpočtu vláken určuje programátor.

**Pointer** je proměnná, jejíž hodnotou je adresa.

**Hyper-threading** - každé vlákno se navenek rozdělí. 16jádro má 16 instrukčních sad.

Co rozumíme pojmem **proces**? OS *alokuje a spravuje paměť*, přidělí *stack* a alespoň jeden *main thread*.

Bloky jsou schedulované pomocí Streaming Multiprocessoru (SM). GPU má pouze omezený počet SM *(NVIDIA GeForce GTX 1650 má 14 SM)*.

<img src="figures/sm-automatic-scalability.png" alt="sm-automatic-scalability" width="400px">

Skrýváním latence (čekání, **latency hiding**) rozumíme zkrácení nečinnosti procesoru. Instrukce mají nějaký čas vykonávání (např. odmocnina nebo modulo je drahá instrukce). Čtení z disku jakožto nejdražší paměťová operace.

32 CUDA vláken běží se stejnou instrukční sadou ve **warpu**.

### 1.1. Model vykonávání (Execution Model)

<img src="figures/execution-model.png" alt="execution-model.png" width="500px">

### 1.2. Amdahlův zákon

Maximální teoretické zrychlení pomocí paralelismu:

$$\boxed{S=\dfrac{T_{old}}{T_{new}}=\dfrac{r_s+r_p}{r_s+\dfrac{r_p}{n}}=\dfrac{1}{r_s+\dfrac{r_p}{n}}}$$

- $S$ - *speed-up*
- $r_s$ - *serial runtime* (čas sekvenčního algoritmu)
- $r_p$ - *parallel runtime* (čas paralelního běhu)
- $n$ - *number of cores*

Příklad: 70 % programu běží sériově. Máme k dispozici 8 jader.

$$ S=\dfrac{1}{0.7+\dfrac{0.3}{8}} \approx 1.35 $$

Algoritmus, kde $r_p=1$, nazýváme "embarrassingly parallel".

### 1.3. N-Body Problem

Výpočet gravitačních interakcí těles, kdy musíme počítat interakce každý s každým (neexistuje matematický model pro $N$ těles).

### 1.4. Boiler Problem

Buď kotel na vodu a dvě kontrolní vlákna. Problém nelze řešit pouze těmito dvěmi vlákny. To, co chce udělat jedno z nich, chce i to druhé. Musí tam být nějaký další prvek, který bude vlákna ovládat (např. semafor, mutex - mutual exclusion).

### 1.5. Dining Philosophers Problem

<img src="figures/dining-philosophers.png" alt="dining-philosophers" width="200px">

### 1.6. nvcc

- Compiler pro rozšíření CUDA.
- Vezme zdroják a rozdělí kód na funkce, které se mají kompilovat pomocí `g++` a `cudapp`.

### 1.7. VisualStudio22

<img src="figures/vs22-setup.png" alt="vs22-setup" width="200px">

- RMB na projekt, `unload`, upravit verze CUDA, `reload`, `rebuild`.

<img src="figures/vs22-properties.png" alt="vs22-properties" width="400px">

- *Compute capability* je dána modelem GPU.

<details><summary> Limity a vlastnosti GPU </summary>

```txt
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

</details>

### 1.8. Jak vybrat N náhodných unikátních čísel z pole?

- Dvě pole: originální a pole náhodných hodnot. Sortuju náhodné pole a podle toho měním pozice v originálním poli. Potom vyberu prvních $N$ hodnot z takto vytvořeného pole.

## 2. Technologie CUDA

<img src="figures/grid-blocks-thread.png" alt="grid-blocks-thread" width="300px">

- Grid se rozpadne na bloky.
- Bloky se rozdělí na warpy. Jeden warp má 32 vláken, které jsou schopny vykonávat SIMT (Single Instruction Multiple Thread).
- `__syncthreads()` - čeká se až všechny warpy dokončí výpočet.

Jeden **streaming multiprocessor (SM) zpracovává jeden blok**. Při dělení problému řádu $N$ obvykle dádá smysl rozdělit úlohu na násobek $|SM|$ bloků, kde $|SM|$ je počet SM. Každý SM má vlastní řídící jednotky, registry, caches a execution pipelines.

**Blok je dělený do warpů po 32 vláknech (thread block).** Vlákna v rámci jednoho warpu jsou **synchronní** (tzn. nedává smysl volat `__syncthreads()`). Warp je fyzicky vykonán paralelně (SIMT) na multiprocesoru.

**Global memory** je hodně, ale je pomalá.

**Karta čte obvykle po 512 B.**

Kernel je funkce (vrácí void), která je spuštěna na `host` a vykonává se asynchronně na `device` (neblokuje `host`). Klíčové slovo `__global__`.

<img src="figures/kernel-grid-block-warp-thread.png" alt="kernel-grid-block-warp-thread" width="300px">

### 2.1. Occupancy

*Occupancy* je poměr průměrného počtu aktivních warpů na SM vůči maximálnímu počtu warpů, který je podporovaný SM.

Vyšší *occupancy* neznámená automaticky vyšší výkon!

**Teoretická occupancy** je dána nastavením kernelu a vlastnostmi GPU. Závisí na:

- velikosti bloku,
- velikosti shared memory (na blok),
- počtem registrů (na blok).

**Dosažená (achieved) occupancy** vychází z měření běhu kernelu. Dosažená occupancy je menší než teoretická occupancy kvůli nevyváženému zatížení bloků (jak uvnitř mezi warpy, tak i mezi bloky).

### 2.2. Práce s vektory

- Zvolím grid, např. $(2,1,1)$.
- Zvolím velikost bloku např. $(128,1,1)$. Není důvod to komplikovat více dimenzemi.

```cpp
unsigned int tid = blockIdx.x * blockDim + threadIdx;
```

### 2.3. Shared memory (SH)

Shared memory (SH) je alokovaná pro každý thread block, všechny vlákna v rámci bloku mají přístup do stejné sdílené paměti. Latence SH je přibližně 10x nižší oproti globální paměti, která není načtená v cache (pokud nedochází k *bank konfliktům* mezi vlákny). Každý SM má k dispozici 64 KB shared memory. SH je paměť s nízkou latencí v blízkosti každého jádra SM (podobně jako  L1 cache CPU).

Multicast přístup k SH znamená, že pokud na stejnou pozici ve stejném čase přistupuje více vláken warpu, tak je provedeno pouze jedno načtení z SH.

Broad cast SH znamená, že více vláken může přistupovat k SH bez omezení výkonu.

- `__syncthreads()`
- `volatile` řekne compileru, že se nemá provádět cache hodnot do registrů (optimizer compileru volí libovolně), používá se při paralelní redukci

SH je uspořádána do banků $32\times 4\text{B}$. Většina 128-bit čtení způsobí **bank conflict**.

#### 2.3.1. Statická shared memory

V kernelu: `__shared__ int vec[256];`

#### 2.3.2. Dynamická shared memory

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

### 2.4. Parallel Reduction

Průchod dat - agregace do jedné hodnoty (min, max, suma atd.)

Optimalizace:

- transfer into a faster memory $\longrightarrow$ hiding memory latency (první krok - vlákna zkopírují hodnoty z globalní paměti do shared memory)
- thread indexing $\longrightarrow$ non-active thread at the end
- coalesced memory $\longrightarrow$ no bank conflicts (viz obrázek)
- algorithm cascading $\longrightarrow$ instruction level parallelism
- warp optimization $\longrightarrow$ avoid synchronization (když zbývá $<32$ hodnot / vláken)

Coalesced memory with thread indexing and shared memory:

<img src="figures/parallel-reduction.png" alt="parallel-reduction" width="550px">

### 2.5. Zarovnaná paměť

```cuda
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
```

Sloupcová vs. řádková matice - do 1D se matice ukládá buď po sloupcích *(column major)* nebo po řádcích *(row major)*.

### 2.6. Bank Conflicts

- Přístup k paměti přes nějaký bank. Lze optimalizovat pomocí vhodného indexování prvků.

### 2.7. Constant Memory

- Klíčové slovo `__constant__`, např. `__constant__ __device__ int myVar;`
- Chová se jako konstanta z pohledu GPU, ale lze měnit z HOST.
- `cudaMemCpyToSymbol` a `cudaMemCpyFromSymbol`
- Přístup k poli uloženém v konstantní paměti se **serializuje**! V tomto případě je lepší SH, která podporuje multicast.

## 3. Texturovací paměť

Obrázek - šířka, výška, formát. Mapování se ukládá do datových struktur.

<img src="figures/2d_texture.png" alt="2d_texture" width="350px">

Blok *read-only* paměti sdílené všemi multi-procesory (SM). Rychlý random-access. Přístupy k texturovací paměti jsou cache-ovány (8 KB na SM).

- 1D - např. gradient - grafická reprezentace nějaké fyzikální veličiny (třeba mapování výšky terénu)
- 2D - obrázek
- 3D - v porovnání s polem 2D textur lze ve 3D provádět interpolaci
- 4D - cube mapa (skybox)

V texturovací jednotce lze nastavit **wrapping** (textura se *opakuje*) nebo **clamping** (přístupy mimo texturu se přesunou na *hranici*).

<img src="figures/wrap_vs_clamp.png" alt="wrap_vs_clamp" width="400px">

Proč se nepoužívá `if` na kontrolu `ouf of bound`? Př. 3K textura, 10k objektů, double buffering, 144 fps $\Rightarrow 9\cdot10^6\cdot10000\cdot2\cdot144\Rightarrow$ obrovské množsví instrukcí a práce pro scheduler.

Kdy se nevyplatí použít texturovací paměť? Kdybychom chtěli data číst pouze jednou poté, kdy je do textury uložíme.

**Mipmapa** je metoda optimalizace textur, kdy se načítájí textury v různých rozlišeních od největší (4k) až do velikosti 1x1 pixel:

<img src="figures/mipmap.png" alt="mipmap" width="250px">

### 3.1. Normálové mapy (Normal Mapping)

<img src="figures/normal-mapping.png" alt="normal-mapping" width="350px">

Příklad normálové mapy (uprostřed) se 3D scénou, ze které byla vypočtena (vlevo), a výsledek při aplikaci na rovný povrch (vpravo).

Normálová mapa je obvykle RGB textura, ve které jsou zakódované souřadnice $x,y,z$. Obvykle převažuje souřadnice $z$, proto normálové mapy bývají do modra (B).

Normálová mapa lze také vypočítat z 2D obrázku pomocí **Sobelova filtru**.

<img src="figures/sobel.png" alt="sobel" width="250px">

- $x$ ... R ... $[-1,1]\Rightarrow[0,255]$
- $y$ ... G ... $[-1,1]\Rightarrow[0,255]$
- $z$ ... B ... $[0,1]\Rightarrow[0,255]$

<img src="figures/normal-map-from-image.png" alt="normal-map-from-image" width="350x">

## 4. OpenGL

- Knihovna, API pro práci s grafickým hardware.
- **Stavový stroj** - všechno je perzistentní dokud to není změněno.
- Používá `main thread`, samotná aplikace musí běžet na jiném vlákně.
- Neumožňuje vytváření oken, není to programovací jazyk.
- Pracuje se v **shaderech**.
- CUDA se přizpůsobuje OpenGL (protože OpenGL je starší). Prvně se vytvoří objekt v OpenGL, ke kterému potom přes pointery přistupujeme z CUDA.

### 4.1. Grafika

Vertex ve 3D je prvek vektorového prostoru $V$, nicméně na obrazovce vidíme jenom projekci do 2D. Manipulujeme s vektory, projektujeme je do jiných vektorových prostorů.

1. **Model-View** matice. Transformace model $\rightarrow$ kamera.
2. **Projekční** matice. Definuje optické vlastnosti pohledu, "vlastnosti čočky" (např. ohnisko kamery). Přesněji definuje **zorný objem** (frustum - komolý hranol), součástí je přední a zadní ořezávací vzdálenost.

   <img src="figures/frustum.png" alt="frustum" width="150px">

3. **NDC** (normalized device coordinates) transformace, která převede komolý hranol (kameru) do normalizované krychle $[-1,-1,-1]$ až $[1,1,1]$. Tzn. transformace kamera $\rightarrow$ normalizovaná krychle.
4. Transformace na rozlišení displeje.

Buffery v počítačové grafice:

1. **Z buffer**: z-ová souřadnice určující, které objekty vidíme (vzdálenost mezi přední a zadní ořezovou rovinou kamery).
2. **Accumulation buffer**: načítání předchozích snímků např. pro *motion-blur*.
3. **Stencil buffer**: maskování.

### 4.2. Double Buffering

- Front buffer, back buffer.
- Jeden buffer ukazuji na obrazovce, druhý upravuji pro další snímek, potom je prohodím.

### 4.3. GLUT library

- Vrstva nad OpenGL pro vytváření oken.

## 5. Atomické intrukce (Atomic Functions)

- Atomická instrukce je **nedělitelná operace**.
- Je HW zaručeno, že atomická operace bude vykonána pouze jedním vláknem.
- SW implementace lze vytvořit pomocí binárního **semaforu**. Jedno vlákno něco dělá, ostatí v nekonečném cyklu čekájí, až se operace dokončí. Toto by bylo velmi pomalé, proto je to řešeno v HW.

### 5.1. CUDA atomické instrukce

- **shared** nebo **global** memory
- atomické operace jsou pomalejší než standardní (load/store)

```cpp
int y = atomicAdd(x, 1) // y = *x; *x = *x + 1
```

Proč to vrací aktuální stav? **Stack unwinding**.

Možné přístupy hledání maxima v poli pomocí atomických operací:

1. Agregace do jedné hodnoty v globální paměti.
2. Agregace do pole v globální paměti (pro každý blok).
3. Agregace do shared memory v rámci jednoho bloku.

## 6. Page Lock Memory

OS stránkuje paměť (kvůli rychlosti, přes cache). Přednačítá data dopředu (kvůli rychlosti). Podobně funguje i GPU.

Princip **page lock memory** je zamknutí paměti na `host` a `device` (RAM a VRAM). OS tím řekneme, aby paměť stránkoval na stejných adresách. Potom GPU nemusí kontrolovat stránky a **memcopy** je rychlejší.

## 7. Unified Memory

Správa paměti je přenechána OS. Nevýhodou je, že nemáme jak zjistit, kdy je paměť na CPU a kdy na GPU. Pokud program navrhneme špatně, tak může docházet k častému kopírovaní dat mezi CPU a GPU (přičemž z pohledu programátora to nemusí na první pohled být viditelné).

## 8. Streamy

- Memory Stream
- File Stream
- Network Stream

Proč mít více streamů? **Skrývání latence** - každá instrukce má nějaký čas vykonávání a my chceme skrývat latenci mezi instrukcemi. Zpracovávání heterogenních úloh. **Task-level parallelism**.

### 8.1. Memory Stream

Co je potřeba k implementaci bufferu?

- **buffer**, třeba lineární (např. **pole**)
- **ukazatel** - pozice v bufferu
- **realokace**
- **pomocné metody**, třeba operátory `<<` a `>>`
- **dealokace**
- vícevláknová implementace: zamykání, přístup, ...

### 8.2. CUDA Streams

CUDA stream je sekvence kernelů nebo CUDA příkazů. Streamy lze využít pro lepší utilizaci `device`.

- `cudaStreamCreate`
- `cudaMemcopyAsync` - asynchronní volání
- `cudaStreamSynchronize` - v asynchronním programování musíme vždy někdy zavolat `synchronize`

Co dělat s více streamy?

1. Paralelismus do **hloubky**
   - Buď $i_n$ $n$-tá instrukce, $s_1$ a $s_2$ streamy.
   - Postup: $i_1$ pro $s_1$, $i_2$ pro $s_1$, ..., $i_n$ pro $s_1$ a až potom $i_1$ pro $s_2$ atd.
2. Paralelismus do **šířky** - všechny streamy mohou hned začít pracovat
   - Buď $i_n$ $n$-tá instrukce, $s_1$ a $s_2$ streamy.
   - Postup: $i_1$ pro $s_1$, $i_1$ pro $s_2$, pak $i_2$ pro $s_1$, $i_2$ pro $s_2$ atd.

Kdy se hodí využít streamy? Např. pro skrytí latence kopírování dat na `device`. Chceme, aby se "překrývalo" vykonávání kernelu a kopírování dat mezi `host` a `device`. Používáme `cudaMemcpyAsync` v kombinaci s `cudaAllocHost` (page-locked memory).

<img src="figures/cuda-stream.png" alt="cuda-stream" width="350px">

## 9. CuBLAS

Operace:

- $V\times V$
- $V\times M$
- $M\times M$

`ld` je *leading dimension* - kolik prvků přeskočit, abych se dostal na další dimenzi (sloupec).

`cublas` apriori používá sloupcové matice. Matice $A^{M\times N} \Rightarrow$ v `cublas` $M$ sloupců dimenze $N$ $\Rightarrow$ `ld=N`.

### 9.1. N-body maticově

- Buď $N$ těles dimenze $D$, buď $a$ a $b$ dvě libovolné tělesa.

$$
\begin{align*}
  \sum\limits_{i=0}^{D-1}(a_i-b_i)^2 &= \sum\limits_{i=0}^{D-1}\left(a_i^2-2a_ib_i+b_i^2\right)=\\
  &=\sum\limits_{i=0}^{D-1}a_i^2-2 \sum\limits_{i=0}^{D-1}a_ib_i + \sum\limits_{i=0}^{D-1}b_i^2\\
  &= A^{.2} -2A^TA + A^{.2}
\end{align*},
$$
kde $A^{.2}$ značí kvadrát elementů matice $A$.

$$
C =
\begin{bmatrix}
\times  & a & b & c\\
a & 0 & d_{a,b} & d_{a,c} \\
b & d_{b,a} & 0  & d_{b,c} \\
c & d_{c,a}  & d_{c,b}  & 0 \\
\end{bmatrix}
$$

<!-- $\Rightarrow A.^2 @ \mathbb{1}^{M,N} - 2 A^TA$ -->

`cublas` má pro násobení matic operaci:

$C \leftarrow \alpha AB+\beta C$

```cpp
// OP_T ... transpose
// OP_N ... normal
cublasSgemm(handler, A, B, C)
```

Řešení v `numpy`:

```python
N,D = A.shape
C = np.zeros((N,N))

C = A**2 @ np.ones((D, N))     + 0 * C
C = -2 * A @ A.T               + 1 * C
C = np.ones((N, D)) @ (A**2).T + 1 * C
```

Druhou mocninu elementů matice a odmocninu spočítáme v kernelu.

## 10. AoS vs. SoA

Array of Structures (AoS):

```c
#define N 10

typedef struct{
    int x;
    int y;
} Entity;

Entity entities[N];
```

Structure of Arrays (SoA):

```c
#define N 10

typedef struct{
    int x[N];
    int y[N];
} Entity;

Entity entity;
```

Kvůli memory coalescing se na GPU většinou preferuje SoA.

## 11. Examples

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

<details><summary> Texture memory </summary>

```cpp
{{#include src/5_texture_memory.cu}}
```

</details>

<details><summary> Texture memory: Normal Map </summary>

```cpp
{{#include src/6_normal_map.cu}}
```

</details>

<details><summary> OpenGL integration in CUDA </summary>

```cpp
{{#include src/7_opengl.cu}}
```

</details>

<details><summary> Atomics </summary>

```cpp
{{#include src/8_atomics.cu}}
```

</details>

<details><summary> CUDA streams </summary>

```cpp
{{#include src/9_cuda_streams.cu}}
```

</details>
