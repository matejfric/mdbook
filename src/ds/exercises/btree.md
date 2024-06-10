# QEP and B+Tree

- [1. Dotazy](#1-dotazy)
  - [1.1. MS SQL Server](#11-ms-sql-server)
  - [1.2. Oracle](#12-oracle)
- [2. QEP - halda a index](#2-qep---halda-a-index)
  - [2.1. MS SQL Server](#21-ms-sql-server)
  - [2.2. Oracle](#22-oracle)
- [3. Index customer\_name\_residence](#3-index-customer_name_residence)
  - [3.1. MS SQL Server](#31-ms-sql-server)
  - [3.2. Oracle](#32-oracle)
- [4. Select MIN/MAX lName and fName](#4-select-minmax-lname-and-fname)
  - [4.1. MS SQL Server](#41-ms-sql-server)
  - [4.2. Oracle](#42-oracle)
- [5. Select MIN/MAX lName and residence](#5-select-minmax-lname-and-residence)
  - [5.1. MS SQL Server](#51-ms-sql-server)
  - [5.2. Oracle](#52-oracle)
    - [5.2.1. MAX](#521-max)
    - [5.2.2. MIN](#522-min)
- [6. Index customer\_lname\_residence](#6-index-customer_lname_residence)
  - [6.1. MS SQL Server](#61-ms-sql-server)
  - [6.2. Oracle](#62-oracle)
    - [6.2.1. MAX](#621-max)
    - [6.2.2. MIN](#622-min)
    - [6.2.3. Opakování dotazu pro selekci na lname, fname, residence](#623-opakování-dotazu-pro-selekci-na-lname-fname-residence)
- [7. Velikosti indexů tabulky Customer](#7-velikosti-indexů-tabulky-customer)
  - [7.1. MS SQL Server](#71-ms-sql-server)
  - [7.2. Oracle](#72-oracle)

## 1. Dotazy

### 1.1. MS SQL Server

```sql
-- lName, fName, residence
--48
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, fName, residence
) counts;

-- lName, fName
--967
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, fName
) counts;

-- lName, residence
--600
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, residence
) counts;
```

### 1.2. Oracle

```sql
--lName, fName, residence
--115
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, fName, residence
) counts;

--lName, fName
--2307
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, fName
) counts;

--lName, residence
--1153
SELECT AVG(record_count) AS mean_count
FROM (
    SELECT COUNT(*) AS record_count
    FROM Customer
    GROUP BY lName, residence
) counts;
```

## 2. QEP - halda a index

- ad1) není vytvořený index pro atributy, podle kterých hledáme záznamy (tzn. je proveden sekvenční průchod haldou)

### 2.1. MS SQL Server

```sql
-- lName, fName, residence
-- 35
SELECT * FROM Customer 
WHERE lName=N'Veselý' AND fName=N'Václav' AND residence='Ostrava'
option (maxdop 1);
```

- Table Scan
- logical reads 1750
- CPU time = 62 ms
- elapsed time = 60 ms

### 2.2. Oracle

```sql
-- lName, fName, residence
-- 58
SELECT * FROM Customer 
WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
```

- TABLE ACCESS FULL
- buffer gets:  524
- cpu_time_ms:  19.793
- elapsed_time_ms:  5.541

<details><summary> Details </summary>

```text
Plan hash value: 2844954298
 
------------------------------------------------------------------------------
| Id  | Operation         | Name     | Rows  | Bytes | Cost (%CPU)| Time     |
------------------------------------------------------------------------------
|   0 | SELECT STATEMENT  |          |    72 |  2952 |   552   (2)| 00:00:01 |
|*  1 |  TABLE ACCESS FULL| CUSTOMER |    72 |  2952 |   552   (2)| 00:00:01 |
------------------------------------------------------------------------------
 
Predicate Information (identified by operation id):
---------------------------------------------------
 
   1 - filter("RESIDENCE"='Ostrava' AND "FNAME"='Alena' AND 
              "LNAME"='Dvořáková')
```

```text
executions:  1
buffer gets:  1961
cpu_time_ms:  13.645
elapsed_time_ms:  17.188
rows_processed:  58
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'
```

</details>

## 3. Index customer_name_residence

### 3.1. MS SQL Server

```sql
CREATE INDEX idx_customer_name_residence ON
    Customer(lName, fName, residence);

BEGIN
        EXEC PrintPages 'CUSTOMER';
        EXEC PrintPagesIndex 'idx_customer_name_residence';
END;
```

Čas vytvoření indexu:

- CPU time = 5675 ms,
- elapsed time = 334 ms.

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Customer | 300000 | 1753 | 13.7 | 1751 | 13.7 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| idx\_customer\_name\_residence | 300000 | 1898 | 14.8 | 1624 | 12.7 |

- index `idx_customer_name_residence` má 1898 bloků (o něco více než celá halda)

Žádná změna:

- Table Scan
- logical reads 1750
- CPU time = 62 ms
- elapsed time = 60 ms

### 3.2. Oracle

```sql
CREATE INDEX idx_customer_name_residence ON Customer(lName, fName, residence);
--855 ms

BEGIN
    PrintPages('CUSTOMER', 'FRI0089');
END;

SELECT segment_name, blocks, bytes/1024/1024 AS size_mb
FROM user_segments
WHERE segment_name = 'IDX_CUSTOMER_NAME_RESIDENCE'; 
```

Čas vytvoření indexu 855 ms. Index `IDX_CUSTOMER_NAME_RESIDENCE` má 1664 bloků, 13 MB (tabulka `CUSTOMER` má 2048 bloků, 16 MB).

- TABLE ACCESS BY INDEX ROWID BATCHED, INDEX RANGE SCAN
- executions:  2
- buffer gets:  63
- cpu_time_ms:  0
- elapsed_time_ms:  1.7235
- rows_processed:  58

<details><summary> Details </summary>

```text
Plan hash value: 2864161656
 
-------------------------------------------------------------------------------------------------------------------
| Id  | Operation                           | Name                        | Rows  | Bytes | Cost (%CPU)| Time     |
-------------------------------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT                    |                             |    72 |  2952 |    73   (0)| 00:00:01 |
|   1 |  TABLE ACCESS BY INDEX ROWID BATCHED| CUSTOMER                    |    72 |  2952 |    73   (0)| 00:00:01 |
|*  2 |   INDEX RANGE SCAN                  | IDX_CUSTOMER_NAME_RESIDENCE |    72 |       |     3   (0)| 00:00:01 |
-------------------------------------------------------------------------------------------------------------------
 
Predicate Information (identified by operation id):
---------------------------------------------------
 
   2 - access("LNAME"='Dvořáková' AND "FNAME"='Alena' AND "RESIDENCE"='Ostrava')
```

```text
executions:  2
buffer gets:  63
cpu_time_ms:  0
elapsed_time_ms:  1.7235
rows_processed:  58
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'
```

</details>

## 4. Select MIN/MAX lName and fName

### 4.1. MS SQL Server

```sql
--similarly for MIN
SELECT lName, fName, COUNT(*) AS record_count
FROM Customer
GROUP BY lName, fName
HAVING COUNT(*) = (
    SELECT MAX(cnt)
    FROM (
        SELECT COUNT(*) AS cnt
        FROM Customer
        GROUP BY lName, fName
    ) as Cc
);
```

```sql
--1217
SELECT * FROM Customer WHERE lName=N'Dvořáková' AND fName=N'Anna'
option (maxdop 1);
```

- Table Scan
- logical reads 1179
- CPU time = 47 ms,  elapsed time = 46 ms

```sql
--752
SELECT * FROM Customer WHERE lName=N'Procházka' AND fName=N'Petr'
option (maxdop 1);
```

- Table Scan
- logical reads 709
- CPU time = 31 ms,  elapsed time = 26 ms

### 4.2. Oracle

```sql
--similarly for MIN
SELECT lName, fName, COUNT(*) AS record_count
FROM CUSTOMER
GROUP BY lName, fName
HAVING COUNT(*) = (
    SELECT MAX(cnt)
    FROM (
        SELECT COUNT(*) AS cnt
        FROM CUSTOMER
        GROUP BY lName, fName
    )
);
```

```sql
SELECT * FROM Customer 
WHERE lName='Kučerová' AND fName='Anna';
```

```text
------------------------------------------------------------------------------
| Id  | Operation         | Name     | Rows  | Bytes | Cost (%CPU)| Time     |
------------------------------------------------------------------------------
|   0 | SELECT STATEMENT  |          |  2812 |   112K|   552   (2)| 00:00:01 |
|*  1 |  TABLE ACCESS FULL| CUSTOMER |  2812 |   112K|   552   (2)| 00:00:01 |
------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  1966
cpu_time_ms:  11.924
elapsed_time_ms:  20.198
rows_processed:  2924
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Kučerová' AND fName='Anna'
```

```sql
explain plan for
SELECT * FROM Customer WHERE lName='Nováková' AND fName='Jana';
```

```text
------------------------------------------------------------------------------
| Id  | Operation         | Name     | Rows  | Bytes | Cost (%CPU)| Time     |
------------------------------------------------------------------------------
|   0 | SELECT STATEMENT  |          |   693 | 28413 |   552   (2)| 00:00:01 |
|*  1 |  TABLE ACCESS FULL| CUSTOMER |   693 | 28413 |   552   (2)| 00:00:01 |
------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  1962
cpu_time_ms:  11.845
elapsed_time_ms:  16.847
rows_processed:  682
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Nováková' AND fName='Jana'
```

## 5. Select MIN/MAX lName and residence

### 5.1. MS SQL Server

```sql
--695
SELECT * FROM Customer WHERE lName=N'Horák' AND residence=N'Přerov'
option (maxdop 1);
```

- Table Scan
- logical reads 1272
- CPU time = 46 ms,  elapsed time = 49 ms

```sql
--519
SELECT * FROM Customer WHERE lName=N'Černá' AND residence=N'Ústí nad Label'
option (maxdop 1);
```

- Table Scan
- logical reads 1654
- CPU time = 62 ms,  elapsed time = 64 ms

### 5.2. Oracle

#### 5.2.1. MAX

```sql
SELECT lName, residence, COUNT(*) AS record_count
FROM CUSTOMER
GROUP BY lName, residence
HAVING COUNT(*) = (
    SELECT MAX(cnt)
    FROM (
        SELECT COUNT(*) AS cnt
        FROM CUSTOMER
        GROUP BY lName, residence
    )
);

explain plan for
SELECT * FROM Customer WHERE lName='Veselá' AND residence='Olomouc';

select * from table(dbms_xplan.display);

set feedback on SQL_ID;
SELECT * FROM Customer WHERE lName='Veselá' AND residence='Olomouc';
set feedback off SQL_ID;

exec PrintQueryStat('cymr1ngr276gx', 2844954298);
```

```text
------------------------------------------------------------------------------
| Id  | Operation         | Name     | Rows  | Bytes | Cost (%CPU)| Time     |
------------------------------------------------------------------------------
|   0 | SELECT STATEMENT  |          |  1351 | 55391 |   552   (2)| 00:00:01 |
|*  1 |  TABLE ACCESS FULL| CUSTOMER |  1351 | 55391 |   552   (2)| 00:00:01 |
------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  1963
cpu_time_ms:  43.681
elapsed_time_ms:  18.581
rows_processed:  1422
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Veselá' AND residence='Olomouc'
```

#### 5.2.2. MIN

```sql
SELECT lName, residence, COUNT(*) AS record_count
FROM CUSTOMER
GROUP BY lName, residence
HAVING COUNT(*) = (
    SELECT MAX(cnt)
    FROM (
        SELECT COUNT(*) AS cnt
        FROM CUSTOMER
        GROUP BY lName, residence
    )
);

explain plan for
SELECT * FROM Customer WHERE lName='Veselá' AND residence='Olomouc';

select * from table(dbms_xplan.display);

set feedback on SQL_ID;
SELECT * FROM Customer WHERE lName='Veselá' AND residence='Olomouc';
set feedback off SQL_ID;

exec PrintQueryStat('cymr1ngr276gx', 2844954298);
```

```text
-------------------------------------------------------------------------------------------------------------------
| Id  | Operation                           | Name                        | Rows  | Bytes | Cost (%CPU)| Time     |
-------------------------------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT                    |                             |   338 | 13858 |   331   (0)| 00:00:01 |
|   1 |  TABLE ACCESS BY INDEX ROWID BATCHED| CUSTOMER                    |   338 | 13858 |   331   (0)| 00:00:01 |
|*  2 |   INDEX RANGE SCAN                  | IDX_CUSTOMER_NAME_RESIDENCE |   338 |       |     4   (0)| 00:00:01 |
-------------------------------------------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  6
cpu_time_ms:  0
elapsed_time_ms:  2.214
rows_processed:  0
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Pospíšilová' AND fName='Jihlava'
```

## 6. Index customer_lname_residence

### 6.1. MS SQL Server

```sql
CREATE INDEX idx_customer_lname_residence 
ON Customer(lName, residence);
```

- CPU time = 2967 ms,  elapsed time = 180 ms

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| idx\_customer\_lname\_residence | 300000 | 1610 | 12.6 | 1354 | 10.6 |

- Plán dotazu se nezmění, DBS stále volí `Table Scan`.

### 6.2. Oracle

```sql
CREATE INDEX idx_customer_lname_residence ON Customer(lName, residence);
--0.537 s
```

#### 6.2.1. MAX

- Stejný výsledek jako bez indexu.

#### 6.2.2. MIN

- Stejný výsledek jako v předchozí úloze, byl použit index.

#### 6.2.3. Opakování dotazu pro selekci na lname, fname, residence

- plán je stejný, použije se index

```sql
explain plan for
SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';

select * from table(dbms_xplan.display);

set feedback on SQL_ID;
SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
set feedback off SQL_ID;
```

```text
-------------------------------------------------------------------------------------------------------------------
| Id  | Operation                           | Name                        | Rows  | Bytes | Cost (%CPU)| Time     |
-------------------------------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT                    |                             |    72 |  2952 |    73   (0)| 00:00:01 |
|   1 |  TABLE ACCESS BY INDEX ROWID BATCHED| CUSTOMER                    |    72 |  2952 |    73   (0)| 00:00:01 |
|*  2 |   INDEX RANGE SCAN                  | IDX_CUSTOMER_NAME_RESIDENCE |    72 |       |     3   (0)| 00:00:01 |
-------------------------------------------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  69
cpu_time_ms:  15.668
elapsed_time_ms:  2.682
rows_processed:  58
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'

```

## 7. Velikosti indexů tabulky Customer

### 7.1. MS SQL Server

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Customer | 300000 | 1753 | 13.7 | 1751 | 13.7 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_Customer\_\_D058768771FC3B31 | 300000 | 673 | 5.3 | 673 | 5.3 |
| idx\_customer\_name\_residence | 300000 | 1898 | 14.8 | 1624 | 12.7 |
| idx\_customer\_lname\_residence | 300000 | 1610 | 12.6 | 1354 | 10.6 |

### 7.2. Oracle

Tabulka CUSTOMER:

- blocks: 2048
- size (MB): 16

| index | blevel | leaf_blocks | blocks | size_MB |
|---|---|---|---|---|
| SYS_C00554393 | 1 | 562 | 640 | 5 |
| IDX_CUSTOMER_NAME_RESIDENCE | 2 | 1559 | 1664 | 13 |
| IDX_CUSTOMER_LNAME_RESIDENCE | 2 | 1310 | 1408 | 11 |
