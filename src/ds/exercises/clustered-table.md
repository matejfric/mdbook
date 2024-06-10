# Clustered Table

- [1. MS SQL Server](#1-ms-sql-server)
  - [1.1. Dotaz se selekcí na PK](#11-dotaz-se-selekcí-na-pk)
  - [1.2. Dotaz se selekcí na name, residence](#12-dotaz-se-selekcí-na-name-residence)
  - [1.3. Vytvoření indexu na name, residence](#13-vytvoření-indexu-na-name-residence)
- [2. Oracle](#2-oracle)
  - [2.1. Dotaz se selekcí na PK](#21-dotaz-se-selekcí-na-pk)
  - [2.2. Dotaz se selekcí na name, residence](#22-dotaz-se-selekcí-na-name-residence)
  - [2.3. Vytvoření indexu na name, residence](#23-vytvoření-indexu-na-name-residence)

## 1. MS SQL Server

```sql
BEGIN
    EXEC PrintPages 'Customer';
    EXEC PrintPagesIndex 'PK__Customer__D058768771FC3B31';
END;
```

| TableName (IndexName)              | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--------------------------------- | :-------- | :--------- | :------------- | :-------- | :------------ |
| Customer                           | 300000    | 1753       | 13.7           | 1751      | 13.7          |
| PK\_\_Customer\_\_D058768771FC3B31 | 300000    | 673        | 5.3            | 673       | 5.3           |

```sql
CREATE CLUSTERED INDEX Customer
ON Customer (idCustomer);

EXEC PrintPagesClusterTable 'Customer';
```

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :-------- | :-------- | :--------- | :------------- | :-------- | :------------ |
| Customer  | 300000    | 2129       | 16.6           | 1820      | 14.2          |

| name                               | level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--------------------------------- | :---- | :---------- | :------------ | :---------------- | :---------------- | :-------- |
| PK\_\_Customer\_\_D058768771FC3B31 | 0     | 384         | 300000        | 8                 | 96.5              | 8.07      |
| PK\_\_Customer\_\_D058768771FC3B31 | 1     | 1           | 384           | 11                | 61.7              | 0         |

```sql
ALTER TABLE Customer REBUILD;
```

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :-------- | :-------- | :--------- | :------------- | :-------- | :------------ |
| Customer  | 300000    | 2009       | 15.7           | 1816      | 14.2          |

### 1.1. Dotaz se selekcí na PK

```sql
SELECT * FROM Customer WHERE idCustomer IN (1234, 42)
option (maxdop 1);
```

- Clustered Index Seek
- Scan count 2, logical reads 6
- CPU time = 0 ms,  elapsed time = 0 ms.

```sql
SELECT * FROM CustomerHeap WHERE idCustomer IN (1234, 42)
option (maxdop 1);
```

- Nested Loops
  - Index Seek
  - RID Lookup
- Scan count 2, logical reads 8
- CPU time = 0 ms,  elapsed time = 0 ms.

### 1.2. Dotaz se selekcí na name, residence

```sql
SELECT * FROM Customer 
WHERE lName=N'Veselý' AND fName=N'Václav' AND residence='Ostrava'
option (maxdop 1);
```

- Clustered Index Scan,
- logical reads 1784,
- CPU time = 63 ms,
- elapsed time = 59 ms,
- 35 rows.

```sql
SELECT * FROM CustomerHeap 
WHERE lName=N'Veselý' AND fName=N'Václav' AND residence='Ostrava'
option (maxdop 1);
```

- Table Scan
- logical reads 1750
- CPU time = 62 ms,
- elapsed time = 60 ms
- 41 rows

### 1.3. Vytvoření indexu na name, residence

```sql
CREATE INDEX idx_customer_name_residence 
ON Customer(lName, fName, residence);
-- CPU time = 8640 ms,  elapsed time = 635 ms

CREATE INDEX idx_customerheap_name_residence 
ON CustomerHeap(lName, fName, residence);
```

- Výsledky se oproti 1.3 nezmění.

## 2. Oracle

```sql
SELECT segment_name, blocks, bytes/1024/1024 AS size_mb
FROM user_segments
WHERE segment_name IN ('CUSTOMER', 'SYS_C00554393');
```

| TableName (Index) | Blocks | SizeMB |
| :---------------- | :----- | :----- |
| CUSTOMER          | 2048   | 16     |
| SYS_C00554393     | 640    | 5      |

```sql
DROP TABLE ORDERITEM;
DROP TABLE "Order";
DROP TABLE CUSTOMER;

create table Customer (
  idCustomer int primary key,
  fName varchar(20) not null,
  lName varchar(30) not null,
  residence varchar(20) not null,
  gender char(1) not null,
  birthday date not null
) ORGANIZATION INDEX;

SELECT index_name, blevel, leaf_blocks
FROM user_indexes
WHERE table_name = 'CUSTOMER';

SELECT segment_name, blocks, bytes/1024/1024 AS size_mb
FROM user_segments
WHERE segment_name IN ('SYS_IOT_TOP_646492');
```

| TableName (Index)  | Blocks | SizeMB |
| :----------------- | :----- | :----- |
| SYS_IOT_TOP_646492 | 1920   | 15     |

```sql
ALTER TABLE CUSTOMER SHRINK SPACE;

SELECT segment_name, blocks, bytes/1024/1024 AS size_mb
FROM user_segments
WHERE segment_name IN ('SYS_IOT_TOP_646492');
```

| TableName (Index)  | Blocks | SizeMB |
| :----------------- | :----- | :----- |
| SYS_IOT_TOP_646492 | 1856   | 14.5   |

### 2.1. Dotaz se selekcí na PK

```sql
SELECT * FROM Customer WHERE idCustomer IN (1234, 42)
```

```text
-----------------------------------------------------------------------------------------
| Id  | Operation          | Name               | Rows  | Bytes | Cost (%CPU)| Time     |
-----------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT   |                    |     2 |    84 |     4   (0)| 00:00:01 |
|   1 |  INLIST ITERATOR   |                    |       |       |            |          |
|*  2 |   INDEX UNIQUE SCAN| SYS_IOT_TOP_646492 |     2 |    84 |     4   (0)| 00:00:01 |
-----------------------------------------------------------------------------------------
 

---- Query Processing Statistics ----
executions:  2
buffer gets:  24.5
cpu_time_ms:  0
elapsed_time_ms:  1.8605
rows_processed:  2
username:  FRI0089
query:  SELECT * FROM Customer WHERE idCustomer IN (1234, 42)
```

```sql
SELECT * FROM CustomerHeap WHERE idCustomer IN (1234, 42)
```

```text
----------------------------------------------------------------------------------------------
| Id  | Operation                    | Name          | Rows  | Bytes | Cost (%CPU)| Time     |
----------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT             |               |    22 |  1452 |    70   (0)| 00:00:01 |
|   1 |  INLIST ITERATOR             |               |       |       |            |          |
|   2 |   TABLE ACCESS BY INDEX ROWID| CUSTOMERHEAP  |    22 |  1452 |    70   (0)| 00:00:01 |
|*  3 |    INDEX UNIQUE SCAN         | SYS_C00570359 |  1478 |       |     5   (0)| 00:00:01 |
----------------------------------------------------------------------------------------------


---- Query Processing Statistics ----
executions:  2
buffer gets:  62
cpu_time_ms:  6.509
elapsed_time_ms:  3.8305
rows_processed:  2
username:  FRI0089
query:  SELECT * FROM CustomerHeap WHERE idCustomer IN (1234, 42)
```

### 2.2. Dotaz se selekcí na name, residence

```sql
SELECT * FROM Customer 
WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
```

```text
-------------------------------------------------------------------------------------------
| Id  | Operation            | Name               | Rows  | Bytes | Cost (%CPU)| Time     |
-------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT     |                    |   115 |  4830 |   498   (2)| 00:00:01 |
|*  1 |  INDEX FAST FULL SCAN| SYS_IOT_TOP_646492 |   115 |  4830 |   498   (2)| 00:00:01 |
-------------------------------------------------------------------------------------------
 
---- Query Processing Statistics ----
executions:  8
buffer gets:  1827
cpu_time_ms:  23.122625
elapsed_time_ms:  28.1995
rows_processed:  68
username:  FRI0089
query:  SELECT * FROM Customer WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'
```

```sql
SELECT * FROM CustomerHeap
WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
```

```text
----------------------------------------------------------------------------------
| Id  | Operation         | Name         | Rows  | Bytes | Cost (%CPU)| Time     |
----------------------------------------------------------------------------------
|   0 | SELECT STATEMENT  |              |     1 |    66 |   553   (2)| 00:00:01 |
|*  1 |  TABLE ACCESS FULL| CUSTOMERHEAP |     1 |    66 |   553   (2)| 00:00:01 |
----------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  1962
cpu_time_ms:  12.215
elapsed_time_ms:  16.819
rows_processed:  51
username:  FRI0089
query:  SELECT * FROM CustomerHeap WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'
```

### 2.3. Vytvoření indexu na name, residence

```sql
CREATE INDEX idx_customer_name_residence 
ON Customer(lName, fName, residence);

CREATE INDEX idx_customerheap_name_residence
ON CustomerHeap(lName, fName, residence);
```

```sql
SELECT * FROM Customer 
WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
```

- Stejný výsledek jako v 2.2.

```sql
SELECT * FROM CustomerHeap
WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava';
```

```text
-----------------------------------------------------------------------------------------------------------------------
| Id  | Operation                           | Name                            | Rows  | Bytes | Cost (%CPU)| Time     |
-----------------------------------------------------------------------------------------------------------------------
|   0 | SELECT STATEMENT                    |                                 |    51 |  3366 |    44   (0)| 00:00:01 |
|   1 |  TABLE ACCESS BY INDEX ROWID BATCHED| CUSTOMERHEAP                    |    51 |  3366 |    44   (0)| 00:00:01 |
|*  2 |   INDEX RANGE SCAN                  | IDX_CUSTOMERHEAP_NAME_RESIDENCE |    51 |       |     3   (0)| 00:00:01 |
-----------------------------------------------------------------------------------------------------------------------

---- Query Processing Statistics ----
executions:  1
buffer gets:  64
cpu_time_ms:  0
elapsed_time_ms:  4.065
rows_processed:  51
username:  FRI0089
query:  SELECT * FROM CustomerHeap WHERE lName='Dvořáková' AND fName='Alena' AND residence='Ostrava'
```
