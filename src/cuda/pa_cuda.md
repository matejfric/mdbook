# PA2

- Datový paralelismus.
- Instrukční paralelismus:
  - využití instrukcí:
    - jedny vlákna chystají

**Logické vlákno** je sled instrukcí. Potřebuju **registry** a nějakou výpočetní jednotku. Běží dokud má instrukce.

**Pointer** je proměnná, jejíž hodnotou je adresa.

Hyper-threading - každé vlákno se jakoby rozdělí.
16jádro má 16 instrukčních sad.

Přerušení výpočtu vláken určuje programátor.

Proces: alokace a management paměti OS, přidělení stacku, alespoň jeden main thread.

Bloky jsou schedulované pomocí Streaming Multiprocessoru (SM). GPU má pouze $n$ SM.

Skrývání latence (čekání, **latency hiding**) - zkrácení nečinnosti procesoru.

- instrukce mají nějaký čas vykonávání
- čtení z disku

CUDA: vlákna se stejnou instrukční sadou běží ve **warpu**.

## Amdahlův zákon

Maximální teoretické zrychlení:

$$ S=\dfrac{1}{r_s+\dfrac{r_p}{n}} $$

- $r_s$...*(serial runtime)* čas sekvenčního algoritmu
- $r_p$...*(parallel runtime)* čas paralelního běhu

Příklad 70 % programu běží sériově. Máme 8 jader.

$$ S=\dfrac{1}{0.7+\dfrac{0.3}{8}} $$

## N-Body Problem

Výpočet gravitačních interakcí těles, kdy musíme počítat interakce každý s každým (neexistuje matematický model pro $n$).

## Boiler Problem

Kotel na vodu a dvě kontrolní vlákna. Problém nelze řešit pouze těmito dvěmi vlákny. To, co chce udělat jedno z nich, chce i to druhé. Musí tam být nějaký další prvek, který bude vlákna ovládat (např. semafor, mutex).

## Dining Philosophers Problem

<img src="figures/dining-philosophers.png" alt="dining-philosophers" width="200px">

## nvcc

- vezme zdroják a rozdělí kód na funkce, které se mají kompilovat g++ a cudapp

## VisualStudio22

<img src="figures/vs22-setup.png" alt="vs22-setup" width="200px">

- RMB na projekt, `unload`, upravit verze CUDA, `reload`, `rebuild`.

<img src="figures/vs22-properties.png" alt="vs22-properties" width="400px">

- *Compute capability* je dána modelem GPU.

## Práce s vektory

- Zvolím grid, např. $(2,1,1)$
- Zvolím velikost bloku např. $(128,1,1)$ - není důvod komplikovat.

```cpp
unsigned int tid = blockIdx.x * blockDim + threadIdx;
```

## Popis CUDA

- Grid se rozpadne na bloky.
- Bloky se rozdělí na warpy. Jeden warp má 32 vláken, které jsou schopny vykonávat SIMT (Single Instruction Multiple Thread).
- `syncthread` - čeká se až se všechny warpy dokončí výpočet.

## Limity GPU

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

- 14 SM
