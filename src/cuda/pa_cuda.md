# PA2

- Datový paralelismus.
- Instrukční paralelismus:
  - využití instrukcí:
    - jedny vlákna chystají

Hyper-threading - každé vlákno se jakoby rozdělí.
16jádro má 16 instrukčních sad.

Přerušení výpočtu vláken určuje programátor.

Proces: alokace a management paměti OS, přidělení stacku, alespoň jeden main thread.

Bloky jsou schedulované pomocí Streaming Multiprocessoru (SM). GPU má pouze $n$ SM.

Skrývání latence (čekání) - zkrácení nečinnosti procesoru.

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
