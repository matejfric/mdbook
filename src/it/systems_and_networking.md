# Počítačové systémy a sítě

- [1. Architektura univerzálních procesorů. Principy urychlování činnosti procesorů](#1-architektura-univerzálních-procesorů-principy-urychlování-činnosti-procesorů)
  - [1.1. Zpracování instrukce a zřetězení (pipelining)](#11-zpracování-instrukce-a-zřetězení-pipelining)
  - [1.2. Problémy zřetězení](#12-problémy-zřetězení)
  - [1.3. Paměťová hierarchie](#13-paměťová-hierarchie)
- [2. Základní vlastnosti monolitických počítačů a jejich typické integrované periférie. Možnosti použití](#2-základní-vlastnosti-monolitických-počítačů-a-jejich-typické-integrované-periférie-možnosti-použití)
  - [2.1. Typické periferní zařízení](#21-typické-periferní-zařízení)
- [3. Protokolová rodina TCP/IP](#3-protokolová-rodina-tcpip)
- [4. Problémy směrování v počítačových sítích. Adresování v IP sítích](#4-problémy-směrování-v-počítačových-sítích-adresování-v-ip-sítích)
- [5. Bezpečnost počítačových sítí s TCP/IP: útoky, paketové filtry, stavový firewall. Šifrování a autentizace, virtuální privátní sítě](#5-bezpečnost-počítačových-sítí-s-tcpip-útoky-paketové-filtry-stavový-firewall-šifrování-a-autentizace-virtuální-privátní-sítě)
- [6. Paralelní výpočty a platformy: Flynnova taxonomie, SIMD, MIMD, SPMD. Paralelismus na úrovni instrukcí, datový a funkční paralelismus. Procesy a vlákna](#6-paralelní-výpočty-a-platformy-flynnova-taxonomie-simd-mimd-spmd-paralelismus-na-úrovni-instrukcí-datový-a-funkční-paralelismus-procesy-a-vlákna)
- [7. Systémy se sdílenou a distribuovanou pamětí: komunikace mezi procesy (souběh, uváznutí, vzájemné vyloučení). Komunikace pomocí zasílání zpráv. OpenMP, MPI](#7-systémy-se-sdílenou-a-distribuovanou-pamětí-komunikace-mezi-procesy-souběh-uváznutí-vzájemné-vyloučení-komunikace-pomocí-zasílání-zpráv-openmp-mpi)
  - [7.1. Model sdílené paměti](#71-model-sdílené-paměti)
    - [7.1.1. OpenMP](#711-openmp)
  - [7.2. Model distribuované paměti](#72-model-distribuované-paměti)
- [8. Paralelní redukce a paralelní scan: principy fungování ve vybrané technologii a příklady užití](#8-paralelní-redukce-a-paralelní-scan-principy-fungování-ve-vybrané-technologii-a-příklady-užití)
  - [8.1. Paralelní redukce](#81-paralelní-redukce)
  - [8.2. Prefix sum (paralelní scan)](#82-prefix-sum-paralelní-scan)
- [9. Konkurentní datové struktury: přehled, blokující a neblokující implementace](#9-konkurentní-datové-struktury-přehled-blokující-a-neblokující-implementace)

## 1. Architektura univerzálních procesorů. Principy urychlování činnosti procesorů

<img src="figures/simplified-pc.png" alt="simplified-pc" width="350px">

> Procesor je *sekvenční obvod*, vstupem jsou instrukce z paměti.

```mermaid
mindmap
  root)Procesor(
    (Definice)
      [Sekvenční obvod]
        (Vstupem jsou instrukce z paměti)
    (Základní části)
      [ALU - Arithmetic Logic Unit]
      [Registry]
        (IP - Instruction Pointer)
      [Řadič - Control Unit]
      [L1 Cache]
    (Kategorie)
      [CISC]
        (Complex Instruction Set Computer)
      [RISC]
        (Reduced Instruction Set Computer)
          [1 stroj. cyklus 1 instrukce]
          [Zřetězené zpracování instrukcí]
    (Urychlování činnosti)
      [Predikce skoku]
        (Jednobitová)
        (Dvoubitová)
        (Statická - kompilátor)
        (Dynamická - za běhu)
      [Paralelní fronta instrukcí]
      [Zřetězení]
        (Datové hazardy)
        (Strukturální hazardy)
        (Řídící hazardy)
      ["Cache (L1,L2,L3)"]
      [Větší počet jader]
      [Vyšší frekvence jader]
      ["SIMD instrukce (AVX)"]
```

### 1.1. Zpracování instrukce a zřetězení (pipelining)

Zpracování instrukce je rozděleno do několika fází a často je možné aby se fáze překrývaly (pokud nejsou instrukce na sobě závislé).

| Krok | Zkratka |Význam                 |
|:------|:-------:|:-----------------|
| 1.   | VI | Výběr Instrukce   |
| 2.   | DE | Dekódování        |
| 3.   | VA | Výpočet Adresy    |
| 4.   | VO | Výběr Operandu    |
| 5.   | PI | Provedení Instrukce |
| 6.   | UV | Uložení Výsledku  |

|CISC|RISC|
|:--:|:--:|
|<img src="figures/cisc-chain.png" alt="cisc-chain" height="125px">|<img src="figures/risc-chain.png" alt="risc-chain" height="125px">|

(V současnosti se nejčastěji používá nějaká kombinace CISC a RISC.)

### 1.2. Problémy zřetězení

- **Datové hazardy** - např. rozpracovaná instrukce potřebuje data předchozí instrukce (může řešit překladač)
  - $a+b+c\Rightarrow$ `ADD a, b; ADD a, c`
- **Strukturální hazardy** - např. omezení sběrnice
- **Řídící hazardy** - např. skok na jinou instrukci (řeší se predikcí skoku nebo paralelní frontou instrukcí)
  - především podmíněné skoky
  - jednobitová predikce
  - dvoubitová predikce (stavový automat)

### 1.3. Paměťová hierarchie

<img src="figures/memory-hierarchy.png" alt="memory-hierarchy" width="600px">

## 2. Základní vlastnosti monolitických počítačů a jejich typické integrované periférie. Možnosti použití

```mermaid
mindmap
  root)Monolitické počítače(
    (Definice)
      [Integrovaný v jediném pouzdře]
      [Tvoří jeden celek - monolit]
      [Někdy také mikropočítač]
    (Komponenty)
      [Procesor CPU]
      [Paměť]
        (RAM)
        (Flash)
      [Periferie]
    (Paměť)
      [Nevolatilní pro program]
        (Flash EEPROM)
      [SRAM pro data]
      [Registry pro data]
    (Architektura)
      [Harvardská]
        (Odděluje paměť dat od paměti programu)
        (Program se nemůže přepsat)
      [Von Neumannova]
    (Typy procesoru)
      [CISC]
        (Complex Instruction Set Computer)
      [RISC]
        (Reduced Instruction Set Computer)
    (Periferní zařízení)
      [LCD displej]
      [LED diody]
        (RGB)
        (PWM - Pulse Width Modulation)
          [Změna jasu LED]
          [D = T · I]
            [D je délka/šířka pulzu]
            [T je perioda]
            [I je intenzita jasu]
      [Tlačítka]
      [Rádio]
      [IR přijímač]
```

> Monolitický počítač je malý počítač *integrovaný v jediném pouzdře* (na jednom čipu). Tvoří jeden celek - monolit. Někdy také *mikropočítač*.
>
> Monolitický počítač obsahuje **procesor** (CPU), **paměť** (RAM, Flash) a **periferie**.

Pro program se používá *nevolatilní* paměť, která zachovává data i po odpojení (např. Flash EEPROM). Pro data se používají registry nebo SRAM paměti. Převážně se tedy používá harvardská architektura, která odděluje paměti pro data od paměti programu (tzn. program se nemůže přepsat).

| Von Neumannova architektura | Harvardská architektura |
|--------------------------|------------------------------|
|<img src="figures/von-neumann.png" alt="von-neumann" height="180px">| <img src="figures/harvard.png" alt="harvard" height="180px">|

Podle typu procesoru rozlišujeme **CISC** (Complex Instruction Set Computer) a **RISC** (Reduced Instruction Set Computer) monolitické počítače.

**Sběrnice** je skupina signálových vodičů, která přenáší data mezi komponenty počítače (např. PCI, USB, I$^2$C).

### 2.1. Typické periferní zařízení

- LCD displej
- LED diody (RGB)
  - PWM (Pulse Width Modulation) - pomocí PWM lze měnit jas LED
    - $D = T \cdot I,$
    - kde $D$ je délka/šířka pulzu *(PWM duty cycle)*, T je perioda (třeba v ms) a $I\in<0,1>$ je intenzita (úroveň jasu)
    - <img src="figures/duty-cycle.gif" alt="duty-cycle" width="250px">
- Tlačítka
- Rádio
- IR přijímač

**Časovač** umožní provádět události v pevných intervalech které se řídí hodinovým signálem (např. polling periferií).

**Watchdog** je časovač který resetuje systém nebo jinak převezme řízení, pokud přestane odpovídat hlavní program (např. zacyklení).

## 3. Protokolová rodina TCP/IP

<img src="figures/tcp-ip.png" alt="tcp-ip" width="800px">

Vrstvy OSI RM:

1. **Fyzická** vrstva
    - fyzický přenos bitů
    - hub (rozbočovač), repeater (opakovač), modemy
    - optické kabely, *Unshielded Twisted Pair (UTP)* kabely (s RJ45 koncovkou)
2. **Spojová** vrstva
    - přenos rámců s **MAC** adresami,
    - **switche** (přepínače)
    - detekce a korekce chyb
3. **Síťová** vrstva
    - *směrování paketů* **routery** (směrovače)
    - **IP** adresy
4. **Transportní** vrstva
    - protokol **TCP** *(Transmission Control Protocol)* - spolehlivý, velký soubor - *segmenty*
    - protokol **UDP** *(User Datagram Protocol)* - nespolehlivý, rychlost, stream - *datagramy*
    - **port**y (trans**port**ní vrstva) `0-65535` $\langle0, 2^{16} - 1\rangle$
5. **Relační** (session) vrstva
    - *dialog mezi účastníky* (udržování a synchronizace komunikace)
6. **Prezentační** vrstva
    - sjednocení formátů dat, kódování
7. **Aplikační** vrstva
    - konkrétní aplikace (prohlížeč, databázový klient)

**ARP** *(Address Resolution Protocol)* - mapování IP adresy na MAC adresu

**ICMP** *(Internet Control Message Protocol)* - `ping`, `traceroute`

**NAT** *(Network Address Translation)* - překlad libovolné IP adresy an jinou IP adresu (nejčastěji privátní na veřejné). Umožňuje aby pod jednou IPv4 adresou bylo více počítačů najednou.

**DNS** *(Domain Name System)* - překlad doménového jména na IP adresu (a naopak)

**DHCP** *(Dynamic Host Configuration Protocol)* - automatické přidělování IP adresy a dalších síťových parametrů (např. DNS serveru) klientům v síti

**TELNET** - nešifrovaný protokol pro vzdálený přístup k počítači (terminálový emulátor)

**SSH** *(Secure Shell)* - šifrovaný protokol pro vzdálený přístup k počítači (terminálový emulátor), šifrování pomocí *asymetrické kryptografie* (veřejný a privátní klíč)

**SMTP** *(Simple Mail Transfer Protocol)* - protokol pro odesílání e-mailů

**POP3** *(Post Office Protocol 3)* - protokol pro stahování a odstraňování e-mailů z poštovního serveru (stáhne na klienta a odstraní ze serveru)

**IMAP** *(Internet Message Access Protocol)* - protokol pro přístup k e-mailům na poštovním serveru (na serveru), stahování kopie

**FTP** *(File Transfer Protocol)* - protokol pro přenos souborů mezi počítači (klient-server/server-klient)

**HTTP** *(Hypertext Transfer Protocol)* - `GET`, `POST`, `PUT`, `DELETE`, ...

Kdysi byla fyzická topologie sítě totožná s logickou, ale dnes se použivají VLAN (virtuální LAN) na spojové vrstvě (L2). Jeden fyzický switch se může chovat jako více logických switchů.

<img src="figures/three-way-handshake.png" alt="three-way-handshake" width="500px">

## 4. Problémy směrování v počítačových sítích. Adresování v IP sítích

```mermaid
mindmap
  root)Směrování(
      (Statické)
        [Manuální konfigurace IP adres]
        [Malé sítě]
      ("Dynamické (protokoly)")
        ["Routing Information Protocol (RIP)"]
          ("Distance Vector Algorithm (DVA)")
          (Nezná rychlost)
          (Hop-count)
        ["Open Shortest Path First (OSPF)"]
          ("Link State Algorithm (LSA)")
          (Ceny linek podle rychlosti)
          (Znalost topologie sítě)
          (Dikstrův algoritmus)

```

**Statické směrování** znamená, že pravidla ve směrovací tabulce jsou spravovány manuálně. Vhodné pouze pro malé sítě.

**Dynamické směrování** se automaticky adaptuje na změny v síti. *Dynamické směrovací protokoly* *(RIP, OSPF)* hledají nejkratší cesty v síti. Samotné směrování provádí *směrovače (routery)* podle *směrovací tabulky*.

- **Routing Information Protocol** *(RIP)* - používá *Distance Vector Algorithm (DVA)*, který nezná rychlost spojení a používá *hop-count* (počet skoků) jako metriku pro určení nejkratší cesty. Router periodicky broadcastem (RIPv1, nebo multicast RIPv2) posílá svoji směrovací tabulku. Nevhodné pro velké sítě.
- **Open Shortest Path First** *(OSPF)* - používá *Link State Algorithm (LSA)*. Router zná topologii sítě a pomocí *Dikstra algoritmu* určí nejkratší cestu na základě ceny spojení podle *přenosové rychlosti*.

|Typ| Cíl/maska | Next hop | Metrika (nižší je lepší) |
|:--:|:---------:|:--------:|:-----------------------:|
|R|10.0.0.0/16 | 172.16.10.1 | 2 |

Typ znamená *typ směrování*:

- C - connected (přímo připojené)
- S - static (staticky konfigurované)
- R - RIP
- O - OSPF

**IP adresa** slouží k identifikaci zařízení v síťové vrstvě TPC/IP modelu. Dnes existují dvě verze adres:

- původní **IPv4** - 32bitové číslo (čtveřice bajtů)
- novější **IPv6** - 128bitové číslo (osmice hexadecimálních číslic)
  - IPv6 poskytuje mnohem větší adresní prostor než IPv4 který už je vyčerpaný.

## 5. Bezpečnost počítačových sítí s TCP/IP: útoky, paketové filtry, stavový firewall. Šifrování a autentizace, virtuální privátní sítě

```mermaid
mindmap
  root )"""Bezpečnost
  TCP/IP""")
    (Útoky)
      [Man-in-the-middle]
      [Denial-of-Service]
      [Spoofing]
      [Sniffing]
    (Firewall)
      [Stavový]
      [Paketový filtr]
    (Šifrování)
      [Symetrické]
      [Asymetrické]
    (VPN)
      [Šifrovaný tunel]
    (Protokoly)
      [TLS]
      [IPsec]
      [SSH]
```

Problém TCP/IP sítí je, že typicky jsou všechna data přenášena nešifrovaně. Data lze nejen přečíst, ale i upravit.

Útoky:

- **Man-in-the-middle (MITM)** - útočník odposlouchává nebo modifikuje komunikaci mezi dvěma stranami. (Lze řešit šifrováním.)
- **Denial-of-Service (DoS)** - cílem je přetížit systém nebo službu velkým množstvím požadavků a způsobit tak její nedostupnost.
- **Spoofing** - útočník se vydává za jiného uživatele (např. podvržením IP adresy).
- **Sniffing** - odposlech síťového provozu.

**Paketové filtry** - druh **firewallu**, který kontroluje pakety na základě pravidel: **IP adresa, port, protokol**.

**Stavový firewall** (stateful) - sleduje **stav spojení**. Bezpečnější než čistý paketový filtr.

**Transport Layer Security** (TLS) je kryptografický protokol pro komunikaci přes počítačovou síť. Používá se v **aplikační vrstvě** TCP/IP. Navíc *Datagram Transport Layer Security (DTLS)* se používá v *transportní vrstvě*. Nahradil zastaralý protokol *Secure Sockets Layer (SSL)*.

**Internet Protocol Security (IPsec)** - sada protokolů **síťové vrstvy**, která obstarává *autentizaci a šifrování paketů* při komunikaci přes Internet Protocol (IP).

**Symetrická kryptografie** - stejný klíč pro šifrování i dešifrování (např. **AES**).

**Asymetrická kryptografie** - šifrování pomocí *veřejného* a *privátního* klíče (např. **RSA**). Veřejný klíč je známý všem, privátní klíč je tajný.

<img src="figures/asymmetric-cryptography.svg" alt="asymmetric-cryptography" width="800px">

- **C**onfidentiality (důvěrnost) - šifrování zprávy
- **I**ntegrity (of data; integrita) - zpráva nebyla změněna během přenosu
- **A**uthenticity (autenticita) - ověření identity odesílatele
- **N**on-repudiation (nepopiratelnost) - odesílatel nemůže popřít, že zprávu odeslal

Asymetrická kryptografie je založena na vynásobení dvou velkých prvočísel, což je rychlý proces, nicméně zpětné hledání těchto dvou čísel (tedy rozklad na prvočísla) je velmi náročný.

**Virtuální privátní sítě** *(VPN – Virtual Private Network)* - vytváří **šifrovaný tunel** mezi klientem a cílovou sítí přes *veřejný internet*. Např. umožňuje bezpečný vzdálený přístup k podnikové síti.

## 6. Paralelní výpočty a platformy: Flynnova taxonomie, SIMD, MIMD, SPMD. Paralelismus na úrovni instrukcí, datový a funkční paralelismus. Procesy a vlákna

*Task-switching - pseudo-paralelismus.*

**Datový paralelismus** - data jsou rozdělena do *bloků* a každý blok je zpracován *procesem/vláknem*.

**Instrukční paralelismus** - využití nezávislých instrukcí - např. jedny vlákna chystají data, další je zpracovávají nebo třeba ukládají.

**Hyper-threading** - každé vlákno se navenek rozdělí. 16jádro má 16 instrukčních sad.

Co rozumíme pojmem **proces**? OS alokuje a spravuje *paměť*, přidělí *zásobník* a alespoň jedno *hlavní jádro (main thread)*.

**Logické vlákno** je *sled instrukcí*. Potřebuju *registry* a nějakou *výpočetní jednotku*. Běží dokud má instrukce. Přerušení výpočtu vláken určuje programátor.

Proč potřebujeme více jader procesoru? **Skrývání latence** - každá instrukce má nějaký čas vykonávání a my chceme skrývat latenci mezi instrukcemi (tzn. zkrátit čas nečinnosti procesoru). Např. odmocnina nebo modulo jsou drahé operace, nejdražší je čtení z disku (nebo dokonce ze vzdáleného disku neno data lake).

**SPMD** (Single Program Multiple Data) - každý proces spouští stejný program, ale na jiných datech. Neexistuje žádný *"hlavní proces"*. Využívá se v **MPI** (Message Passing Interface) - compiler `mpicc` (C), `mpicxx` (C++). Typicky se používá v HPC (High Performance Computing) - na superpočítačích.

```mermaid
mindmap
  root)Flynnova taxonomie(
      (SISD)
        [Skalární instrukce]
      (SIMD)
        ["Vektorové instrucke (AVX)"]
        ["SIMT (GPU)"]
      (MIMD)
        [Multi-core CPU]
        ["MIMD (jádra) + SIMD (AVX)"]
      (MISD)
        %% https://doi.org/10.1145/358234.358246
        [Palubní počítač raketoplánu Discovery]
      (SPMD)
        [HPC]
        [MPI]
```

|| Single Data | Multiple Data |
|:--:|:--:|:--:|
|Single<br>Instruction|<img src="figures/SISD.svg" alt="SISD https://en.wikipedia.org/wiki/User:Cburnett" width="250px"> |<img src="figures/SIMD.svg" alt="SIMD https://en.wikipedia.org/wiki/User:Cburnett" width="250px"> |
|Multiple<br>Instruction|<img src="figures/MISD.svg" alt="MISD https://en.wikipedia.org/wiki/User:Cburnett" width="250px"> | <img src="figures/MIMD.svg" alt="MIMD https://en.wikipedia.org/wiki/User:Cburnett" width="250px">|

<img src="figures/flynn-tax.png" alt="flynn-tax" width="400px">

Skalární přístup (SISD)

```c
float a[8], b[8], c[8];
for (int i = 0; i < 8; i++)
    c[i] = a[i] + b[i];
```

**Vektorové instrukce** (SIMD) - instrukce, které provádějí stejnou operaci na více datech najednou (např. AVX)

```c
// x86-64 AVX2
#include <immintrin.h>

// SIMD (Single Instruction, Multiple Data) 
// AVX (Advanced Vector Extensions)
__m256 vec_a = _mm256_loadu_ps(a);
__m256 vec_b = _mm256_loadu_ps(b);
__m256 vec_c = _mm256_add_ps(vec_a, vec_b);
_mm256_storeu_ps(c, vec_c);
```

## 7. Systémy se sdílenou a distribuovanou pamětí: komunikace mezi procesy (souběh, uváznutí, vzájemné vyloučení). Komunikace pomocí zasílání zpráv. OpenMP, MPI

- komunikace sdílením stavu (mutexy)
- komunikace zasíláním zpráv (kanály, MPI)

### 7.1. Model sdílené paměti

<img src="figures/shared-memory.png" alt="shared-memory" width="250px">

Procesy *sdílejí adresní prostor*, kde můžou *asynchronně* číst a zapisovat.

**Souběh** *(race condition)* nastane např. když dvě nebo více vláken přistupuje současně ke stejnému místu v paměti, alespoň jedno z nich zapisuje a vlákna nepoužívají synchronizaci k řízení svého přístupu (toto je porušení základního pravidla Rustu - *aliasing + mutabilita*). K souběhu může dojít také pokud výsledek programu závisí na pořadí vykonávání vláken.

Přístup do sdílené paměti je synchronizován pomocí **vzájemného vyloučení** *(**mut**ual **ex**clusion - mutex)*, což lze řešit například pomocí *binárního semaforu* neboli zámku (hodnota `0/1` - `locked/unlocked`).

**Uváznutí** *(deadlock)* nastane, např. když dva procesy čekají navzájem na uvolnění zámku nebo když jeden proces se pokusí získat dvakrát stejný zámek.

```rust
use std::sync::Mutex;

fn drop_late(m: &Mutex<Option<u32>>) {
    if let Some(v) = m.lock().unwrap().as_ref() {
        println!("The Option contains a value {v}!");
        // Deadlock (attempts to acquire a second lock on the same mutex)
        m.lock().unwrap();
    }
}
```

#### 7.1.1. OpenMP

<details><summary> Nastavení ve Visual Studio </summary>

1. Right-click your project > `Properties`.
2. `Configuration Properties > C/C++ > Language`
3. Set OpenMP Support to `Yes (/openmp)`.
4. `Configuration Properties > C/C++ > Command Line`
5. In the `Additional Options` box at the bottom, add: `-openmp:experimental` (OpenMP 5.0+).

</details>

```cpp
#include <iostream>
#include <omp.h>

int main() {
    int num_threads = omp_get_max_threads();
    std::cout << "Max available threads: " << num_threads << std::endl;

    // Paralelní blok
    #pragma omp parallel
    {
        // Každé vlákno provede tento blok
        std::cout << "Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
    }
    
    #pragma omp parallel
    {
        #pragma omp critical
        // Tento blok je chráněn vzájemným vyloučením.
        // Pouze jedno vlákno může provést tento blok najednou.
        std::cout << "Thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << std::endl;
    }

    // #pragma omp barrier 
    // Čekání na všechny vlákna, před pokračováním dál (fork-join).
    // (Implicitně na konci každého paralelního bloku.)

    return 0;
}
```

### 7.2. Model distribuované paměti

<img src="figures/distributed-memory.png" alt="distributed-memory" width="300px">

Každý proces má vlastní data a adresní prostor. Procesy komunikují pomocí **zasílání zpráv** (např. pomocí MPI). MPI je standard pro komunikaci mezi procesy v distribuovaném systému. Usnadňuje implementaci, protože řeší detaily komunikace mezi procesy - výběr nejrychlejší cesty (shared memory, síť, ...), pořadí zásílaných zpráv, jistota, že zpráva dorazila, atd. Implementace standardu MPI je např. OpenMPI. Existují bindingy pro různé jazyky (C, C++, Fortran, Python, ...).

MPI navíc implementuje standardní komunikační rutiny jako:

- **point-to-point**
- **broadcast** (*one-to-many*, stejná zpráva pro všechny)
- **scatter** (*one-to-many*, různé zprávy pro různé procesy)
- **gather** (*many-to-one*, různé zprávy pro jeden proces, opak scatter)
- **reduce** (*many-to-one*, agregace dat z více procesů do jednoho procesu, např. `sum`, `max`, `min`)

<img src="figures/mpi-communication-routines.png" alt="mpi-communication-routines" width="400px">

```bash
# Start MPI program with 4 processes on a single computer
mpiexec -n 4 ./my_program
```

## 8. Paralelní redukce a paralelní scan: principy fungování ve vybrané technologii a příklady užití

### 8.1. Paralelní redukce

Průchod dat - **agregace** do jedné hodnoty (`min`, `max`, `sum` atd.)

<img src="figures/parallel-reduction.png" alt="parallel-reduction" width="600px">

```cpp
#include <iostream>
#include <omp.h>  // OpenMP

int main() {
    const int N = 10;
    int sum = 0;

    // Paralelní redukce
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += i;
        std::cout << "Thread " << omp_get_thread_num() << " added " << i << std::endl;
    }
    std::cout << "Sum MIMD = " << sum << std::endl;

    sum = 0;
    // Redukce pomocí vektorových instrukcí (AVX2)
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += i;
    }
    std::cout << "Sum SIMD = " << sum << std::endl;

    // a kombinace... MIMD + SIMD
    // #pragma omp parallel for simd reduction(+:sum)
    
    return 0;
}
```

### 8.2. Prefix sum (paralelní scan)

> **Hillis-Steele** (Stride to $n$):
>
> <img src="figures/prefix-sum-hillis-steele.png" alt="prefix-sum-hillis-steele" width="400px">
>
> Zápisem $\oplus_{1:4}$ rozumíme $a_1\oplus a_2\oplus a_3\oplus a_4$, kde $\oplus$ je libovolná **asociativní binární operace** (např. `+`, `*`, `XOR`).

> **Up Sweep & Down Sweep**
>
> - [Guy E. Blelloch - Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
> - Modré šipky značí sčítání, šedé přesun, červená reset na nulu.
> - (Algoritmus pracuje in-place s jedním polem stejně jako Hillis-Steele)
>
> <img src="figures/prefix-sum-up-down-sweep.png" alt="prefix-sum-up-down-sweep" width="400px">

```cpp
#include<iostream>
#include<chrono>

#define N 10

using namespace std;

int main() {
    int a[N], simd_scan[N], scan_a;
    for (int i = 0; i < N; i++) {
        a[i] = i;
        simd_scan[i] = 0;
    }
    scan_a = 0;
    #pragma omp simd reduction(inscan, +:scan_a)
    for (int i = 0; i < N; i++) {
        scan_a += a[i];
        #pragma omp scan inclusive(scan_a)
        simd_scan[i] = scan_a;
    }
    std::cout << "SIMD Scan Output:\n";
    for (int i = 0; i < N; i++)
        std::cout << simd_scan[i] << "\t";
    return 0;
}
```

## 9. Konkurentní datové struktury: přehled, blokující a neblokující implementace

Naivní použití standardních datových struktur může vést k souběhu (race condition). Konkurentní datové struktury obvykle používají nějaké prostředky pro synchronizaci - vzájemné vyloučení, atomické operace. Např. `queue.Queue` (MPMC kanál) v Pythonu je konkurentní datová struktura, která používá zámky pro synchronizaci přístupu k frontě.

Problém blokující implementace je, že vlákno čeká na zámek - *spin / busy wait* - se 100% vytížením jádra. Pro neblokující implementaci I/O operací můžeme použít např. `epoll` (Linux) pro monitorování více file descriptorů najednou (stačí jednoho vlákno). Alternativou je asynchronní programování - vlákna se neblokují, ale čekají na události, např. `async/await` v Rustu.

```rust
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:6379").await.unwrap();

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        // A new task is spawned for each inbound socket. The socket is
        // moved to the new task and processed there.
        tokio::spawn(async move {
            process(socket).await;
        });
    }
}
```

**MPSC** channel:

```rust
use std::sync::mpsc::channel;
use std::thread;

let (sender, receiver) = channel();

// Spawn off an expensive computation
thread::spawn(move || {
    sender.send(expensive_computation()).unwrap();
});

// Do some useful work for awhile

// Let's see what that answer was
// (blocking)
println!("{:?}", receiver.recv().unwrap());
```

**SPSC** channel `tokio::sync::oneshot:channel()` - např. pro signál k vypnutí serveru.