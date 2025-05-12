# Databázové systémy

- [1. Relační datový model, SQL; funkční závislosti, dekompozice a normální formy](#1-relační-datový-model-sql-funkční-závislosti-dekompozice-a-normální-formy)
  - [1.1. Funkční závislosti](#11-funkční-závislosti)
  - [1.2. Normální formy](#12-normální-formy)
- [2. Transakce, zotavení, log, ACID, operace COMMIT a ROLLBACK; problémy souběhu, řízení souběhu: zamykání, úroveň izolace v SQL](#2-transakce-zotavení-log-acid-operace-commit-a-rollback-problémy-souběhu-řízení-souběhu-zamykání-úroveň-izolace-v-sql)
- [3. Procedurální rozšíření SQL, PL/SQL, T-SQL, triggery, funkce, procedury, kurzory, hromadné operace](#3-procedurální-rozšíření-sql-plsql-t-sql-triggery-funkce-procedury-kurzory-hromadné-operace)
- [4. Fyzická implementace databázových systémů: tabulka (halda a shlukovaná tabulka) a index typu B-strom, materializované pohledy, rozdělení dat](#4-fyzická-implementace-databázových-systémů-tabulka-halda-a-shlukovaná-tabulka-a-index-typu-b-strom-materializované-pohledy-rozdělení-dat)
- [5. Plán vykonávání dotazů, logické a fyzické operace, náhodné a sekvenční přístupy, ladění vykonávání dotazů](#5-plán-vykonávání-dotazů-logické-a-fyzické-operace-náhodné-a-sekvenční-přístupy-ladění-vykonávání-dotazů)
- [6. Stránkování výsledku dotazu, komprimace tabulek a indexů, sloupcové a řádkové uložení tabulek](#6-stránkování-výsledku-dotazu-komprimace-tabulek-a-indexů-sloupcové-a-řádkové-uložení-tabulek)
- [7. CAP teorém, NoSQL DBS, BASE, replikace, MongoDB, CRUD operace](#7-cap-teorém-nosql-dbs-base-replikace-mongodb-crud-operace)

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

## 7. CAP teorém, NoSQL DBS, BASE, replikace, MongoDB, CRUD operace

**CRUD** - `Create`, `Read`, `Update`, `Delete`
