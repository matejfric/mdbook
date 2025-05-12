# Databázové systémy

- [1. Relační datový model, SQL; funkční závislosti, dekompozice a normální formy](#1-relační-datový-model-sql-funkční-závislosti-dekompozice-a-normální-formy)
  - [1.1. Funkční závislosti](#11-funkční-závislosti)
  - [1.2. Normální formy](#12-normální-formy)
- [2. Transakce, zotavení, log, ACID, operace COMMIT a ROLLBACK; problémy souběhu, řízení souběhu: zamykání, úroveň izolace v SQL](#2-transakce-zotavení-log-acid-operace-commit-a-rollback-problémy-souběhu-řízení-souběhu-zamykání-úroveň-izolace-v-sql)
- [3. Procedurální rozšíření SQL, PL/SQL, T-SQL, triggery, funkce, procedury, kurzory, hromadné operace](#3-procedurální-rozšíření-sql-plsql-t-sql-triggery-funkce-procedury-kurzory-hromadné-operace)
- [4. Fyzická implementace databázových systémů: tabulka (halda a shlukovaná tabulka) a index typu B-strom, materializované pohledy, rozdělení dat](#4-fyzická-implementace-databázových-systémů-tabulka-halda-a-shlukovaná-tabulka-a-index-typu-b-strom-materializované-pohledy-rozdělení-dat)
- [5. Plán vykonávání dotazů, logické a fyzické operace, náhodné a sekvenční přístupy, ladění vykonávání dotazů](#5-plán-vykonávání-dotazů-logické-a-fyzické-operace-náhodné-a-sekvenční-přístupy-ladění-vykonávání-dotazů)
- [6. Stránkování výsledku dotazu, komprimace tabulek a indexů, sloupcové a řádkové uložení tabulek](#6-stránkování-výsledku-dotazu-komprimace-tabulek-a-indexů-sloupcové-a-řádkové-uložení-tabulek)
  - [6.1. Stránkování výsledku dotazu](#61-stránkování-výsledku-dotazu)
  - [6.2. Komprimace](#62-komprimace)
    - [6.2.1. Prefixová komprimace](#621-prefixová-komprimace)
    - [6.2.2. Slovníková komprimace](#622-slovníková-komprimace)
  - [6.3. Řádkové uložení dat](#63-řádkové-uložení-dat)
  - [6.4. Sloupcové uložení dat](#64-sloupcové-uložení-dat)
- [7. CAP teorém, NoSQL DBS, BASE, replikace, MongoDB, CRUD operace](#7-cap-teorém-nosql-dbs-base-replikace-mongodb-crud-operace)
  - [7.1. BASE](#71-base)
  - [7.2. MongoDB](#72-mongodb)

## 1. Relační datový model, SQL; funkční závislosti, dekompozice a normální formy

**Relační datový model** je způsob uložení dat v databázi pomocí relací $R\subseteq \text{atributy} \times \text{n-tice}$. V praxi často uvažujeme tabulky s uspořádanými řádky a sloupci (relace $R$ nedefinuje uspořádání). Výběr atributů je **projekce** a výběr řádku je **selekce**.

<img src="figures/relational-model.svg" alt="relational_model" width="375px">

**SQL** *(Structured Query Language)* je **deklarativní jazyk** (tzn. specifikuje co udělat, ne jak to udělat) pro provádění dotazů nad relačními databázovými systémy. Zahrnuje příkazy které můžeme rozdělit do tří kategorií:

1. **DQL - Data Query Language** - dotazování, založeno na relační algebře `SELECT`
2. **DML - Data Manipulation Language** - úprava obsahu tabulek `INSERT`, `UPDATE`, `DELETE`
3. **DDL - Data Definition Language** - úprava struktury tabulky `CREATE`, `ALTER`, `DROP`

### 1.1. Funkční závislosti

> Buď $R$ relace a buď $X,Y\subseteq R$ množiny atributů. Řekneme, že $Y$ **funkčně závisí** na $X$ (značíme $X\to Y$) pokud platí pro každé dvě $n$-tice:
>
> $$(\forall t_1,t_2\in R)\colon \left[t_1[X]=t_2[X]\right] \implies \left[t_1[Y]=t_2[Y]\right].$$
>
> (Pokud mají dvě n-tice stejnou hodnotu pro atributy $X$, musí mít stejnou hodnotu i pro atributy $Y$. `False => True`, takže každý atribut závisí na atributu s unikátními hodnotami.)

**Armstrongovy axiomy**. Bud $X,Y,Z$ množiny atributů relace $R$.

1. **Reflexivita**: Pokud $Y\subseteq X$, pak $X\to Y$.
2. **Rozšíření**: Pokud $X\to Y$, pak $XZ\to YZ$ pro libovolné $Z$.
3. **Transitivita**: Pokud $X\to Y$ a $Y\to Z$, pak $X\to Z$.

Navíc lze dokázat platnost:

1. **Dekompozice**: Pokud $X\to YZ$, pak $X\to Y$ a $X\to Z$. Důkaz:
   1. $X\to YZ$ (dáno)
   2. $YZ\to Y$ (rozšíření)
   3. $[X\to YZ \land YZ\to Y] \implies X\to Y$ (transitivita)
2. **Sjednocení**: Pokud $X\to Y$ a $X\to Z$, pak $X\to YZ$. Důkaz:
   1. $X\to Y$ a $X\to Z$ (dáno)
   2. $X\to Z \implies X\to XZ$ (rozšíření o $X$)
   3. $X\to Y \implies XZ\to YZ$ (rozšíření o $Z$)
   4. $X\to YZ$ (tranzitivita)

### 1.2. Normální formy

**Klíč** je nejmenší podmnožina atributů, která jednoznačně identifikuje jinou množinu atributů.

Proč normální formy? Konzistence dat, odstranění redundance. Např. `Nakup(JmenoZakaznika, Produkty, Cena)` není v žádné NF. Normální formy jsou mírou kvality návrhu databáze.

1. **1NF** - **atributy musí být atomické** (nedělitelné) - např. `Produkt = "jablko, hruška"` nebo `Adresa = "Ostrava, Hlavní třída 25"` není 1NF. Takové atributy je třeba rozdělit na více atributů nebo tabulek s vazbou 1:N.
2. **2NF** - **každý neklíčový atribut** je **závislý** na ***celém* klíči**, ne jen na jeho části. Důsledek: atributy v tabulce mezi sebou nemají vztah M:N (`zákazník x produkt` je M:N) (+1NF).
3. **3NF** - **nesmí existovat závislosti mezi atributy, které nejsou součástí klíče**. Data nejsou tranzitivně závislá (+2NF).
4. **BCNF** (Boyce-Codd) - pro každou funkční závislost `X -> Y` platí, že `X` je klíč tabulky (+3NF).

**Uzávěr** množiny atributů $X$ (značíme $X+$) je množina všech atributů, které podle atributů $X$ můžeme dohledat ("podle šipek").

**Dekompozice** je proces rozdělení tabulky na několik menších, aby byly splněny podmínky normálních forem.

<details><summary> Příklad </summary>

Tabulka `Kniha`. Předpokládáme jednoho autora a *kandidátní klíč* `(Autor, Název)`.

| Autor | Název | Formát | Strany | Cena | Nakladatelství | Země nakladatelství |
|--------|-------|--------|--------|------|-----------------|---------------------|
| Codd | Databázové systémy | kniha | 300 | 499 | Cambridge | UK |
| Codd | Databázové systémy | e-kniha | 300 | 399 | Cambridge | UK |
| Boyce | Databázové systémy 2 | e-kniha | 400 | 299 | Harvard | USA |

Cena závisí na formátu, tzn. existuje závislost `Formát -> Cena`, která není závislá na klíči. Vytvoříme tabulku `FormatCena(*Název, Formát, Cena)` a tabulku `Kniha(*Název, Autor, Formát, Strany, Nakladatelství, Země nakladatelství)`, kde klíč je `Název`. Tím získáme 2NF.

Dále existuje tranzitivní závislost `Název -> Nakladatelství -> Země nakladatelství`. Vytvoříme tabulku `Nakladatelství(*Nakladatelství, Země)`. Tím získáme 3NF.

</details>

Poznámky:

- V **konceptuálním modelu** (entity-relation diagramy) nezáleží na volbě databázového systému.
- Databáze je **konzistentní** právě tehdy, když jsou splněny všechny **integritní omezení**.
- **Databázový systém / Systém pro řízení báze dat (SŘBD)** je aplikace poskytující rozhraní pro vytvoření databáze a pro komunikaci s databází.
- **Databáze** je (multi)množina vzájemně propojených dat, které jsou uloženy v databázovém systému.

## 2. Transakce, zotavení, log, ACID, operace COMMIT a ROLLBACK; problémy souběhu, řízení souběhu: zamykání, úroveň izolace v SQL

> **Transakce** je sekvence příkazů která převede databázi z jednoho konzistentního stavu do druhého
konzistentního stavu. Transakce je atomická operace, buď jsou provedeny všechny příkazy transakce, nebo žádný.

Relační databáze musí splňovat vlastnosti kterým souhrnně říkáme **ACID**:

- **A**tomicity (atomičnost) — operace se provede buď celá nebo vůbec.
- **C**onsistency (konzistence) — databáze musí být vždy v konzistentním stavu (musí být splněny všechny integritní omezení).
- **I**solation (izolovanost) — souběžné transakce musí být izolované (nesmí se navzájem ovlivňovat).
- **D**urability (trvalost) — po dokončení transakce je stav databáze trvale uložen.

## 3. Procedurální rozšíření SQL, PL/SQL, T-SQL, triggery, funkce, procedury, kurzory, hromadné operace

## 4. Fyzická implementace databázových systémů: tabulka (halda a shlukovaná tabulka) a index typu B-strom, materializované pohledy, rozdělení dat

## 5. Plán vykonávání dotazů, logické a fyzické operace, náhodné a sekvenční přístupy, ladění vykonávání dotazů

## 6. Stránkování výsledku dotazu, komprimace tabulek a indexů, sloupcové a řádkové uložení tabulek

### 6.1. Stránkování výsledku dotazu

> Někdy nepotřebujeme v aplikaci zobrazit všechny výsledky dotazu najednou. Např. tabulka obsahuje 100 000 záznamů, ale v UI se uživateli zobrazuje jen jedna stránka záznamů, např. 100.

1. **Cachování na úrovni aplikačního serveru** - je vhodné (bezproblémové) pouze v případě **statických** nebo téměř statických dat.
2. **Stránkování na úrovni DBS a jeho podpora v ORM** - např. metoda třídy `Student`: `Student.Select(loIndex, hiIndex)`

Dotaz se stránkováním (1. stránka):

```sql
SELECT * FROM Customer
WHERE residence = 'Ostrava'
ORDER BY lname, idCustomer
OFFSET 0
ROWS FETCH NEXT 100 ROWS ONLY;
```

### 6.2. Komprimace

- Ke komprimaci v DBS se obvykle využívají spíše jednodušší, starší a rychlé algoritmy, např. **RLE (Run-Length-Encoding)**.
- Upřednostňujeme **propustnost** *(rychlost komprese/dekomprese)* před **kompresním poměrem** *(kolik se uvolní místa)*.
- Kódy proměnné délky *(Eliasovy, Fibonacciho, atd.)* se spíše nepoužívají, protože jsou pomalejší.
- Používá se např. **prefixová komprimace klíčů** B-stromu. Využívá se především u složených klíčů s větším počtem atributů.
- Kdy se vyplatí vyšší komprimace i za cenu pomalejší rychlosti dotazu *(vyšší komprimační poměr, nižší propustnost)*? Např. pro **historická data**, které se nepoužívají příliš často.

#### 6.2.1. Prefixová komprimace

1. Pro každý sloupec je určena hodnota, kterou lze použít ke zmenšení úložného prostoru pro hodnoty v každém sloupci.
2. Tyto hodnoty jsou uloženy jako metadata *(compression information - CI)* za hlavičkou tabulky.
3. Shodující se prefixy jsou nahrazeny referencemi do *CI*.

<img src="../ds/figures/prefix-compression-before.png" alt="prefix-compression-before" width="200px">
<img src="../ds/figures/prefix-compression-after.png" alt="prefix-compression-after" width="200px">

#### 6.2.2. Slovníková komprimace

Slovníková komprimace je aplikována po prefixové. Není omezena jen na jednotlivé sloupce, funguje nad celou tabulkou. Zjednodušeně se kódují opakující se sekvence (kód je umístěn do *CI*).

<img src="../ds/figures/dict-compression.png" alt="dict-compression" width="200px">

### 6.3. Řádkové uložení dat

- V blocích haldy jsou data uložena po záznamech, mluvíme o **řádkovém uložení** **(rowstore)**.
- Řádkové uložení je **výhodné** v případě **projekce na všechny nebo větší počet atributů**:
  - `SELECT * FROM Customer` – sekvenční průchod haldou.
  - `SELECT * FROM Customer WHERE idCustomer=1` – bodový dotaz v indexu, přístup k záznamu v haldě.
- Naopak je řádkové uložení **nevýhodné** v případě **projekce na nízký počet atributů**:
  - `SELECT AVG(sysdate - birthday) FROM Customer` – sekvenční průchod tabulkou a počítání součtu věku, bloky ovšem obsahují i hodnoty ostatních atributů.

### 6.4. Sloupcové uložení dat

- Pokud v dotazech pracujeme jen s několika **málo atributy** (reálné tabulky mohou mít desítky atributů), můžeme uvažovat o tzv. **sloupcovém uložení dat (columnstore)**.
- Jednotlivé hodnoty neobsahují identifikátor řádku (klíč, RID atd.). **Záznamy jsou rekonstruovány podle pořadí hodnot ve sloupci!**
- Sloupcové uložení je výhodné zejména, pokud dotazy pracují s **malým počtem atributů** při **sekvenčním průchodu** tabulky (typicky **agregace**). Je to tedy "opačný" koncept ke konceptu indexu - sekvenční průchod menším objemem dat při nízké selektivitě dotazů.
- Je výhodné data ve sloupcích třídit? Kvůli komprimaci a vykonávání některých dotazů ano, nicméně, *kvůli rekonstrukci záznamů, musíme zachovat stejné pořadí* v jednotlivých sloupcích, případně k setříděným sloupcům uložit klíč (nebo RID).
- Interně může být každý sloupec reprezentovaný jednou haldou.
- Při sloupcovém uložení můžeme dosáhnout **vyššího kompresního poměru**.

## 7. CAP teorém, NoSQL DBS, BASE, replikace, MongoDB, CRUD operace

> Mějme **distribuovaný DBS** (DDBS) rozložený na více počítačích v síti (tzv. **uzlech**).
>
> **CAP teorém** (**Brewerův teorém**) tvrdí, že pro *distribuovaný DBS* není možné dodržet více než dvě vlastnosti z těchto tří:
>
> - **Konzistence (Consistency)**: každé čtení vrátí buď výsledek posledního zápisu, nebo chybu.
> - **Dostupnost (Availability)**: každé čtení vrátí výsledek (nikdy ne chybu), nemusí se ale jednat o výsledek posledního zápisu.
> - **Odolnost k přerušení sítě (Partition tolerance)**: systém pracuje dál i v případě, že dojde ke ztrátě nebo zdržení libovolného počtu zpráv mezi uzly.
>
> **Při výskytu přerušení systém volí mezi dostupností a konzistencí**, není možné zajistit oboje. Dostupnost a konzistenci je možné zajistit jen v případě neexistence přerušení.

V případě výskytu přerušení sítě, systém musí vybírat mezi dvěma akcemi:

1. Zrušit operaci a tak snížit dostupnost, ale zajistit konzistenci. V případě výskytu přerušení, systém vrátí chybu.
2. Vykonat operaci a tak zachovat dostupnost, ale riskovat nekonzistenci. V případě výskytu přerušení, systém vrátí dostupnou verzi výsledku, nemusí se tedy jednat o výsledek posledního zápisu.

Typicky rozlišujeme dva typy DBS na základě CAP teorému:

- **CP** - konzistence a odolnost vůči přerušení (relační DBS s ACID)
- **AP** - dostupnost a odolnost vůči přerušení (NoSQL DBS s BASE)

> NoSQL databázové systémy jsou označení poměrně široké třídy DBS, které (spíše):
>
> - Nepoužívají relační datový model,
> - Nepoužívají SQL,
> - Nepoužívají transakční model ACID,
> - Používají model **klíč-hodnota** (např. JSON dokument) nebo komplikovanější datový model (**strom** pro XML dokumenty nebo **graf**),
> - Nejsou konkurenční k relačním DBS, jsou určeny pro jiné problémy.

- Oracle a MS SQL taky umožňují ukládání grafů, XML dokumentů apod. Nicméně pracují s těmito daty pomocí modelu ACID.
- Nelze tvrdit, že NoSQL je lepší než transakční model. Záleží na aplikaci.

### 7.1. BASE

**Případná konzistence (Eventual consistency)** je model konzistence používaný v **distribuovaných** databázových systémech k dosažení vysoké dostupnosti.

Případná konzistence znamená, že pokud provedeme nějaké zápisy a systém bude pracovat **dostatečně dlouho bez dalších zápisů, data se nakonec zkonsolidují**: další čtení pak budou vracet stejnou hodnotu (posledního zápisu).

Systémy založené na **případné konzistenci** jsou často klasifikovány jako systémy s vlastností **BASE**:

- **V podstatě dostupné (Basically-available)**: Čtení a zápis jsou **maximálně dostupné** s použitím všech uzlů sítě, ale **nemusí být konzistentní**, což znamená, že **čtení nemusí vracet poslední zápis**.
- **Soft-state**: Není garantována konzistence. Po zápisech a nějakém čase chodu systému existuje pouze určitá pravděpodobnost konvergence dat $\Rightarrow$ případná konzistence.
- **Případná konzistence (Eventual consistency)**.

> **Replikace dat** znamená, že data jsou uložena v několika kopiích (replikách) na uzlech DDBS systému. Cílem je zvýšení dostupnosti.

### 7.2. MongoDB

- **Dokumentová databáze** typu **klíč-hodnota**, kde dokumentem je formát podobný **JSON** (**BSON**).
- Dokument je záznam v dokumentové databázi.
- V JSON dokumentech nepoužíváme dekompozici na entitní typy: ukládáme entity v jednom dokumentu.
- Neexistuje schéma databáze (můžeme ale použít, pokud chceme).
- Položky v dokumentu odpovídají roli sloupců v SQL databázi a lze je indexovat pro zvýšení rychlosti vyhledávání.
- **Nevýhoda**: **redundance**, není možná validace dat dle schématu.
- **Výhoda**: **jednodušší dotazování**, ptáme se na dokument, **nepoužíváme operaci spojení** pro spojování entit.

**CRUD** - `Create`, `Read`, `Update`, `Delete`
