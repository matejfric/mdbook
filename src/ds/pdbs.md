# PDBS

- [1. Testovací databáze ProductOrderDb](#1-testovací-databáze-productorderdb)
- [2. Tabulka typu halda (heap table)](#2-tabulka-typu-halda-heap-table)
  - [2.1. Mazání záznamů](#21-mazání-záznamů)
  - [2.2. Vkládání záznamů](#22-vkládání-záznamů)

## 1. Testovací databáze ProductOrderDb

<img src="figures/test-db.png" alt="test-db" width="700px">

## 2. Tabulka typu halda (heap table)

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
