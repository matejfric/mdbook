# PDBS

- [1. Testovací databáze ProductOrderDb](#1-testovací-databáze-productorderdb)
- [2. Tabulka typu halda (heap table)](#2-tabulka-typu-halda-heap-table)
  - [2.1. Mazání záznamů](#21-mazání-záznamů)
  - [2.2. Vkládání záznamů](#22-vkládání-záznamů)
- [3. Indexy v DBS](#3-indexy-v-dbs)
  - [3.1. B-strom](#31-b-strom)
  - [3.2. B+strom](#32-bstrom)
  - [3.3. Rozsahový dotaz](#33-rozsahový-dotaz)
  - [3.4. Index](#34-index)
  - [3.5. PL/SQL](#35-plsql)
    - [3.5.1. Zjištění indexů vytvořených pro tabulku](#351-zjištění-indexů-vytvořených-pro-tabulku)
    - [3.5.2. Počet bloků indexu](#352-počet-bloků-indexu)
    - [3.5.3. Výška B+stromu](#353-výška-bstromu)
    - [3.5.4. Další statistiky](#354-další-statistiky)
  - [3.6. SQL Server](#36-sql-server)
    - [3.6.1. Zjištění indexů vytvořených pro tabulku](#361-zjištění-indexů-vytvořených-pro-tabulku)
    - [3.6.2. Počet bloků indexu](#362-počet-bloků-indexu)
    - [3.6.3. Výška B+stromu](#363-výška-bstromu)
    - [3.6.4. Další statistiky](#364-další-statistiky)

## 1. Testovací databáze ProductOrderDb

<img src="figures/test-db.png" alt="test-db" width="700px">

## 2. Tabulka typu halda (heap table)

> Lineární složitost vyhledávání a neprovádění fyzického mazání záznamů.

- Základní datová struktura pro tabulky relačního datového modelu je **tabulka typu halda** (stránkované pole, resp. stránkovaný seznam).
- Záznamy jsou uloženy ve stránkách/blocích o velikosti nejčastěji 8 kB (používají se násobky alokační jednotky systému, nejčastěji 2kB).
- Vyhledávání je *sekvenční* $\mathcal{O}(n)$.

<details><summary> Příklad: Počet stránek heap table </summary>

(Oracle) Počet stránek:

```sql
SELECT COUNT(*) FROM Customer;
SELECT blocks FROM users_segments
WHERE segment_name = 'CUSTOMER';
```

(Oracle) Počet *využitých* stránek (hlubší rozbor):

```sql
CREATE OR REPLACE PROCEDURE PrintPages (
    ptablename VARCHAR,
    pusername VARCHAR
)
AS
    blocks NUMBER;
    bytes NUMBER;
    unused_blocks NUMBER;
    unused_bytes NUMBER;
    expired_blocks NUMBER;
    expired_bytes NUMBER;
    unexpired_blocks NUMBER;
    unexpired_bytes NUMBER;
    mega NUMBER := 1024.0 * 1024.0;
BEGIN
    dbms_space.unused_space(pusername, ptablename, 'TABLE', blocks, bytes, unused_blocks, unused_bytes, expired_blocks, expired_bytes, unexpired_blocks, unexpired_bytes);
    dbms_output.put_line('blocks: ' || blocks);
    dbms_output.put_line('size (MB): ' || (bytes / mega));
    dbms_output.put_line('used blocks: ' || (blocks - unused_blocks));
    dbms_output.put_line('size used (MB): ' || ((bytes / mega) - (unused_bytes / mega)));
    dbms_output.put_line('unused blocks: ' || unused_blocks);
    dbms_output.put_line('size unused (MB): ' || (unused_bytes / mega));
END;
/

EXEC PrintPages('CUSTOMER', 'KRA28');

SELECT blocks FROM user_segments
WHERE segment_name = 'CUSTOMER';
```

(MS SQL Server) Počet *využitých* stránek (hlubší rozbor):

```sql
CREATE OR ALTER PROCEDURE PrintPages 
    @tableName VARCHAR(30),
    @indexId INT
AS
BEGIN
    SELECT
        t.NAME AS TableName,
        p.rows AS RowCounts,
        SUM(a.total_pages) AS TotalPages,
        ROUND(CAST(SUM(a.total_pages) * 8 AS FLOAT) / 1024, 1) AS TotalPages_MB,
        SUM(a.used_pages) AS UsedPages,
        ROUND(CAST(SUM(a.used_pages) * 8 AS FLOAT) / 1024, 1) AS UsedPages_MB
    FROM sys.tables t
    INNER JOIN sys.indexes i ON t.OBJECT_ID = i.object_id
    INNER JOIN sys.partitions p ON i.object_id = p.OBJECT_ID AND i.index_id = p.index_id
    INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
    WHERE t.NAME = @tableName AND p.index_id = @indexId
    GROUP BY t.Name, p.Rows
    ORDER BY t.Name;
END;
GO

CREATE OR ALTER PROCEDURE PrintPagesHeap 
    @tableName VARCHAR(30)
AS
BEGIN
    EXEC PrintPages @tableName, 0;
END;
GO

EXEC PrintPagesHeap 'Customer';
EXEC PrintPagesHeap 'OrderItem';
```

</details>

### 2.1. Mazání záznamů

- Záznamy v tabulce nejsou nijak uspořádány. Mazání po každé operaci delete, by v nejhorším případě, znamenalo přesouvání $n$ záznamů v hladě.
- Operace mazání pouze *označí záznam jako smazaný*! Záznam musíme prvně najít, proto složitost $\mathcal{O}(n)$.

<details><summary> Příklad: Fyzické mazání záznamů heap table </summary>

Oracle:

```sql
ALTER TABLE OrderItem ENABLE ROW MOVEMENT;
ALTER TABLE OrderItem SHRINK SPACE;
```

MS SQL Server:

```sql
ALTER TABLE <TableName> REBUILD;
```

</details>

### 2.2. Vkládání záznamů

Při vkládání je záznam umístěn na první nalezenou volnou pozici v tabulce (časová složitost $\mathcal{O}(n)$) nebo na konec pole (složitost $\mathcal{O}(1)$).

Teoretická složitost vkládání do haldy je $\mathcal{O}(1)$, ale:

- Pro *primární klíče* a *jedinečné atributy (unique)* je nutné kontrolovat jedinečnost hodnot atributů.
- Referenční integrita - DBS musí kontrolovat hodnoty pro cizí klíče, zda se záznam vyskytuje v odkazované tabulce.

V haldě mají tyto kontroly složitost v $\mathcal{O}(n)$. Halda je základní úložiště dat pro tabulku, potřebujeme ale také úložiště s lepší časovou složitostí základních operací.

## 3. Indexy v DBS

### 3.1. B-strom

<img src="figures/b-tree.png" alt="b-tree" width="500px">

### 3.2. B+strom

$B^+$-strom řádu $C$ má vlastnosti:

- Vnitřní/listový uzel/stránka obsahuje $C-1$ klíčů, vnitřní uzel obsahuje $C$ ukazatelů na dětské uzly.
- **Stránkovatelný** (srovnáme s binárním stromem): C je nastaveno dle velikosti stránky např. 8kB.
- **Vyvážený**: vzdálenost od všech listů ke kořenovému uzlu je stejná.
- **Výška** $h$ je vzdálenost od kořene k listu (počet hran): $h\approx \lceil \log C(n) \rceil$ $\Rightarrow$ maximální počet klíčů $N = C^{h+1} − 1$.
- **Mazání, vkládání** a dotaz na jeden klíč (**bodový dotaz**) mají **časovou složitost** $\mathcal{O}(\log(n))$.
- Počet uzlů/stránek (**IO cost**), které je nutné projít při bodovém dotazu, je $h + 1$.
- Klíče jsou uloženy pouze v interních uzlech. Oproti $B$-stromu má hodnoty pouze v listových uzlech.
- Listové uzly jsou propojené, což pomáhá v rozsahových dotazech.

<img src="figures/b+tree.png" alt="b+tree" width="500px">

Pokud chceme vložit klíč do listového uzlu, který je plný, dojde k operaci **štěpení (split)**. V původním uzlu se ponechá 50% položek, do nově vytvořeného uzlu se přesune 50% položek. Důsledkem je **50% využití stránek** $B$-stromu $\Rightarrow$ $B$-strom je tedy cca 2x větší než halda.

<img src="figures/btree-split.png" alt="btree-split" width="300px">

### 3.3. Rozsahový dotaz

`between 42 and 420`

1. Bodový dotaz pro nižší hodnotu v rozsahu $(42)$.
2. Porovnávání dalších klíčů ve stránce dokud klíč $\leq 420$.
3. Po porovnání všech klíčů stránky je načtena další listová stránka (Každá listová stránka $B^+$-stromu obsahuje odkaz na následující listovou stránku).

<img src="figures/b+tree-range-query.png" alt="b+tree-range-query" width="350px">

<img src="figures/btree-range-query.png" alt="btree-range-query" width="350px">

- IO cost = $h + b$
    1. $h$: bodový dotaz (minimum).
    2. $b$: počet prohledávaných listových uzlů.
- Nejhorší případ $\mathcal{O}(n)$ (průchod všech listových stránek).
- Nejlepší případ: IO cost = $h+1$.
- Sousední listové stránky na *disku* jsou umístěny daleko od sebe, při načítání stránek z disku dochází k **náhodným přístupům** (o 2-3 řády pomalejší), proto je někdy pro plán vykonávání zvolen sekvenční průchod haldou.
  - Náhodný přístup v hlavní paměti není tak problematický díky **cache CPU**.

> Pokud jsou stránky $B$-stromu umístěny v hlavní paměti, i pro vyšší $b$ je použit rozsahový dotaz nad $B$-stromem. Pokud jsou stránky umístěny na disku, i pro relativně nízké $b$, DBS použije sekvenční průchod v haldě.

### 3.4. Index

> Index v DBS je většinou implementován jako $B$-strom.

Index neobsahuje celé záznamy, ale pouze:

- **setřízené hodnoty indexovaných atributů (klíč)**.
- **ROWID** (**RID** v SQL Server), které odkazuje na záznam (řádek) v haldě.

Klíč a ROWID pak nazýváme **položkou** uzlu B-stromu.

Typy indexů:

1. Automaticky vytvořený index:
   - Je vytvořen pro primární klíče a jedinečné atributy (unique), když je úložištěm tabulky halda (heap).
2. Ručně vytvořený index:

```sql
CREATE INDEX <index name>
ON <table name>(<list of attributes>)
```

- Klíč B-stromu obsahuje hodnoty atributů z `<list of attributes>`.

Základní schéma úložiště pro tabulku:

<img src="figures/basic-storage-schema.png" alt="basic-storage-schema" width="700px">

> **ROWID (RID)** je nejčastěji 4–10 bytová hodnota skládající se z **čísla bloku** a **pozice záznamu v haldě**.

Proč potřebujeme číslo bloku? Protože bez toho bychom museli prvně najít správnou stránku DB a nebylo by tam žádné zlepšení oproti sekvenčnímu vyhledávání.

Proč DBS nepoužívá paměťový ukazatel? Protože bloky mohou být umístěny na **disku**!

Proč automaticky vytvářené indexy?

- Rychlejší **kontrola jedinečnosti** hodnoty PK (resp. unique atribut).
- Rychlejší **bodové dotazy** pro PK (resp. unique atribut).
- Rychlejší kontrola **referenční integrity** při mazání záznamu tabulky s PK, na který ukazuje FK (jiné tabulky).

DBS nám **ne**umožní automaticky vytvořený index zrušit.

### 3.5. PL/SQL

#### 3.5.1. Zjištění indexů vytvořených pro tabulku

```sql
SELECT index_name 
FROM user_indexes 
WHERE table_name = 'CUSTOMER';

-- Output: INDEX_NAME    SYS_C00552552
```

Tzn. pro primární klíč `idCustomer` je vytvořen index `SYS_C00552552`, B+strom, kde položka obsahuje hodnotu `idCustomer` a `ROWID`, které ukazuje na kompletní záznam do haldy.

#### 3.5.2. Počet bloků indexu

Počet alokovaných bloku (odhad - maximální hodnota):

```sql
SELECT blocks 
FROM user_segments 
WHERE segment_name = 'CUSTOMER';
-- BLOCKS 2048
```

```sql
SELECT blocks 
FROM user_segments 
WHERE segment_name = 'SYS_C00552552';
-- BLOCKS 640
```

<details><summary> PL/SQL procedura PrintPagesUnusedSpace </summary>

```sql
CREATE OR REPLACE PROCEDURE PrintPagesUnusedSpace (
    ptablename VARCHAR,
    pusername VARCHAR,
    ptype VARCHAR
)
AS
    freeblocks NUMBER;
    blocks NUMBER;
    bytes NUMBER;
    unusedblocks NUMBER;
    unusedbytes NUMBER;
    expiredblocks NUMBER;
    expiredbytes NUMBER;
    unexpiredblocks NUMBER;
    unexpiredbytes NUMBER;
    mega NUMBER := 1024.0 * 1024.0;
BEGIN
    dbms_space.unused_space(
        pusername,
        ptablename,
        ptype,
        blocks,
        bytes,
        unusedblocks,
        unusedbytes,
        expiredblocks,
        expiredbytes,
        unexpiredblocks,
        unexpiredbytes
    );

    dbms_output.put_line('blocks: ' || blocks || ', ' || CHR(9) || 'size (MB): ' || (bytes / mega));
    dbms_output.put_line('used blocks: ' || (blocks - unusedblocks) || ', ' || CHR(9) || 'size (MB): ' || ((bytes / mega) - (unusedbytes / mega)));
    dbms_output.put_line('unused blocks: ' || unusedblocks || ', ' || CHR(9) || 'size (MB): ' || (unusedbytes / mega));
    dbms_output.put_line('expired blocks: ' || expiredblocks || ', ' || CHR(9) || 'unexpired blocks: ' || unexpiredblocks);
END;

EXEC PrintPagesUnusedSpace(’CUSTOMER’, ’KRA28’, ’TABLE’);
--blocks: 2048, size (MB): 16
--used_blocks: 2048, size (MB): 16
--unused_blocks: 0, size (MB): 0
--expired_blocks: 7, unexpired_blocks: 128
```

</details>

<details><summary> PL/SQL procedura PrintPagesSpaceUsage </summary>

```sql
CREATE OR REPLACE PROCEDURE PrintPagesSpaceUsage (
    ptablename VARCHAR,
    pusername VARCHAR,
    ptype VARCHAR
)
AS
    unformatted_blocks NUMBER;
    unformatted_bytes NUMBER;
    fs1_blocks NUMBER;
    fs1_bytes NUMBER;
    fs2_blocks NUMBER;
    fs2_bytes NUMBER;
    fs3_blocks NUMBER;
    fs3_bytes NUMBER;
    fs4_blocks NUMBER;
    fs4_bytes NUMBER;
    full_blocks NUMBER;
    full_bytes NUMBER;
BEGIN
    dbms_space.space_usage(
        pusername,
        ptablename,
        ptype,
        unformatted_blocks,
        unformatted_bytes,
        fs1_blocks,
        fs1_bytes,
        fs2_blocks,
        fs2_bytes,
        fs3_blocks,
        fs3_bytes,
        fs4_blocks,
        fs4_bytes,
        full_blocks,
        full_bytes,
        null
    );

    dbms_output.put_line('unformatted blocks: ' || unformatted_blocks);
    dbms_output.put_line('fs1 blocks (0 to 25% free space): ' || fs1_blocks);
    dbms_output.put_line('fs2 blocks (25 to 50% free space): ' || fs2_blocks);
    dbms_output.put_line('fs3 blocks (50 to 75% free space): ' || fs3_blocks);
    dbms_output.put_line('fs4 blocks (75 to 100% free space): ' || fs4_blocks);
    dbms_output.put_line('full blocks: ' || full_blocks);
END;

EXEC PrintPagesSpaceUsage(’CUSTOMER’, ’KRA28’, ’TABLE’);

--unformatted_blocks: 62
--fs1_blocks (0 to 25% free space): 0
--fs2_blocks (25 to 50% free space): 1
--fs3_blocks (50 to 75% free space): 0
--fs4_blocks (75 to 100% free space): 31
--full_blocks: 1914
```

</details>

#### 3.5.3. Výška B+stromu

```sql
SELECT index_name, blevel, leaf_blocks
FROM user_indexes
WHERE table_name = 'CUSTOMER';
```

```text
INDEX_NAME          BLEVEL LEAF_BLOCKS
--------------- ---------- -----------
SYS_C00552552            1         562
```

- Jeden kořen a 562 listových uzlů
- IO cost bodového dotazu $h+1=2$.

#### 3.5.4. Další statistiky

Využijeme příkaz `ANALYZE INDEX <index_name> VALIDATE STRUCTURE;`, který naplní tabulku `index_stats`.

```sql
ANALYZE INDEX SYS_C00552552 VALIDATE STRUCTURE;

SELECT height - 1 AS h,
       blocks,
       lf_blks AS leaf_pages,
       br_blks AS inner_pages,
       lf_rows AS leaf_items,
       br_rows AS inner_items,
       pct_used --využití stránek (až 100 % díky optimalizace)
FROM index_stats
WHERE name = 'SYS_C00552552';
```

| H   | BLOCKS | LEAF_PAGES | INNER_PAGES | LEAF_ITEMS | INNER_ITEMS | PCT_USED |
|-----|--------|------------|-------------|------------|-------------|----------|
| 1   | 640    | 562        | 1           | 300000     | 561         | 100      |

### 3.6. SQL Server

#### 3.6.1. Zjištění indexů vytvořených pro tabulku

```sql
CREATE OR ALTER PROCEDURE PrintIndexes
    @tableName VARCHAR(30)
AS
BEGIN
    SELECT i.name AS indexName
    FROM sys.indexes i --systémový katalog
    INNER JOIN sys.tables t ON t.object_id = i.object_id --systémový katalog
    WHERE t.name = @tableName AND i.name IS NOT NULL;
END;
GO

EXEC PrintIndexes 'Customer';
-- indexName    PK__Customer__D058768742B8AE8D
```

#### 3.6.2. Počet bloků indexu

```sql
CREATE OR ALTER PROCEDURE PrintPagesIndex
    @indexName VARCHAR(30)
AS
BEGIN
    SELECT
        i.name AS IndexName,
        p.rows AS ItemCounts,
        SUM(a.totalpages) AS TotalPages,
        ROUND(CAST(SUM(a.totalpages) * 8 AS FLOAT) / 1024, 1) AS TotalPages_MB,
        SUM(a.usedpages) AS UsedPages,
        ROUND(CAST(SUM(a.usedpages) * 8 AS FLOAT) / 1024, 1) AS UsedPages_MB
    FROM
        sys.indexes i
    INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
    INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
    WHERE
        i.name = @indexName
    GROUP BY
        i.name, p.Rows
    ORDER BY
        i.name;
END;
GO

EXEC PrintPagesHeap 'Customer'; --kapitola 2 - halda
--TableName   RowCounts   TotalPages  TotalPages_MB   UsedPages   UsedPages_MB
--Customer    300000      1753        13.7            1751        13.7

EXEC PrintPagesIndex 'PK__Customer__D058768742B8AE8D';
--IndexName                           ItemCounts   TotalPages  TotalPages_MB   UsedPages   UsedPages_MB
--PK__Customer__D058768742B8AE8D      300000      673         5.3             673         5.3
```

#### 3.6.3. Výška B+stromu

```sql
SELECT i.name, s.index_depth - 1 AS height,
    SUM(s.page_count) AS page_count
FROM sys.dm_db_index_physical_stats (DB_ID(N'kra28'),
    OBJECT_ID(N'Customer'), NULL, NULL, 'DETAILED') AS s
JOIN sys.indexes AS i
ON s.object_id = i.object_id AND s.index_id = i.index_id
WHERE name = 'PK__Customer__D058768742B8AE8D'
GROUP BY i.name, s.index_depth;

--name                            height   page_count
--PK__Customer__D058768742B8AE8D       2          672
```

- IO cost bodového dotazu $h+1=3$.

#### 3.6.4. Další statistiky

```sql
SELECT
    s.indexlevel AS level,
    s.pagecount,
    s.recordcount,
    s.avgrecordsizeinbytes AS avg_record_size,
    ROUND(s.avgpagespaceusedinpercent, 1) AS page_utilization,
    ROUND(s.avgfragmentationinpercent, 2) AS avg_frag
FROM
    sys.dm_db_index_physical_stats(
        DB_ID(N'kra28'), OBJECT_ID(N'Customer'),
        NULL, NULL, 'DETAILED') AS s
JOIN
    sys.indexes i ON s.object_id = i.object_id AND s.index_id = i.index_id
WHERE
    name = 'PKCustomerD058768742B8AE8D';
```

| level | page_count | record_count | avg_record_size | page_utilization | avg_frag |
|-------|------------|--------------|-----------------|------------------|----------|
|   0   |    669     |    300000    |       16        |       99.7       |   0.45   |
|   1   |     2      |     669      |       11        |       53.7       |   100    |
|   2   |     1      |      2       |       11        |        0.3       |    0     |

- `level`: 0 - listové úrovně, 2 - kořenová úroveň
- `page_count`: B-strom obsahuje pouze 3 vnitřní uzly.
- `record_count`: vnitřní uzly obsahují jen 771 klíčů

Využití listových stránek je téměř 100%, zřejmě důsledek optimalizace při vkládání inkrementovaných hodnot PK. Pokud je hodnota nízká, použijeme `ALTER INDEX REBUILD`.

Průměrná fragmentace uzlů `avg_frag` je míra shody logického a fyzického pořadí stránek. Pokud je hodnota nízká, použijeme `ALTER INDEX REORGANIZE`.
