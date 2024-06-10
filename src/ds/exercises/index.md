# B+Tree and Index

- [1. Velikost indexu pro primární klíč](#1-velikost-indexu-pro-primární-klíč)
  - [1.1. PL/SQL](#11-plsql)
    - [1.1.1. CUSTOMER](#111-customer)
    - [1.1.2. PRODUCT](#112-product)
    - [1.1.3. STORE](#113-store)
    - [1.1.4. Order](#114-order)
    - [1.1.5. ORDERITEM](#115-orderitem)
  - [1.2. T-SQL](#12-t-sql)
    - [1.2.1. Customer](#121-customer)
    - [1.2.2. Product](#122-product)
    - [1.2.3. Store](#123-store)
    - [1.2.4. Order](#124-order)
    - [1.2.5. OrderItem](#125-orderitem)
- [2. B+strom](#2-bstrom)
  - [2.1. PL/SQL](#21-plsql)
  - [2.2. T-SQL](#22-t-sql)
    - [2.2.1. Customer](#221-customer)
    - [2.2.2. Product](#222-product)
    - [2.2.3. Store](#223-store)
    - [2.2.4. Order](#224-order)
    - [2.2.5. OrderItem](#225-orderitem)
- [3. Vytvoření indexu](#3-vytvoření-indexu)
  - [3.1. PL/SQL](#31-plsql)
  - [3.2. T-SQL](#32-t-sql)

## 1. Velikost indexu pro primární klíč

Proč je relativní velikost vůči haldě v případě indexu pro primární klíč tabulky `OrderItem` větší než u ostatních automaticky vytvořených indexů? Primární klíč `OrderItem` je složen ze dvou atributů - `idOrder`, `idProduct`.

### 1.1. PL/SQL

| Table     | Blocks | Size (MB) |
|-----------|--------|-----------|
| CUSTOMER  | 2048   | 16        |
| PRODUCT   | 640    | 5         |
| STORE     | 8      | 0.0625    |
| Order     | 2048   | 16        |
| ORDERITEM | 16384  | 128       |

| Table     | Index         | Blocks | Size (MB) |
|-----------|---------------|--------|-----------|
| CUSTOMER  | SYS_C00554393 | 640    | 5         |
| PRODUCT   | SYS_C00554397 | 256    | 2         |
| STORE     | SYS_C00554400 | 8      | 0.0625    |
| Order     | SYS_C00554404 | 1024   | 8         |
| ORDERITEM | SYS_C00554411 | 22528  | 176       |

#### 1.1.1. CUSTOMER

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 2048  |
| size (MB)                   | 16    |
| used_blocks                 | 2048  |
| size (MB)                   | 16    |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 62    |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 1  |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 32 |
| full_blocks                 | 1913  |

SYS_C00554393

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 640   |
| size (MB)                   | 5     |
| used_blocks                 | 640   |
| size (MB)                   | 5     |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 14    |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 38 |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 0  |
| full_blocks                 | 570   |

#### 1.1.2. PRODUCT

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 640   |
| size (MB)                   | 5     |
| used_blocks                 | 640   |
| size (MB)                   | 5     |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 78    |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 1  |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 40 |
| full_blocks                 | 503   |

SYS_C00554397

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 256   |
| size (MB)                   | 2     |
| used_blocks                 | 256   |
| size (MB)                   | 2     |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 0     |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 49 |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 0  |
| full_blocks                 | 195   |

#### 1.1.3. STORE

| Attribute                   | Value  |
|-----------------------------|--------|
| blocks                      | 8      |
| size (MB)                   | 0.0625 |
| used_blocks                 | 8      |
| size (MB)                   | 0.0625 |
| unused_blocks               | 0      |
| size (MB)                   | 0      |
| expired_blocks              | 7      |
| unexpired_blocks           | 8      |
| unformatted_blocks          | 0      |
| fs1_blocks (0 to 25% free space) | 1 |
| fs2_blocks (25 to 50% free space) | 0 |
| fs3_blocks (50 to 75% free space) | 0 |
| fs4_blocks (75 to 100% free space) | 1 |
| full_blocks                 | 3      |

SYS_C00554400

| Attribute                   | Value  |
|-----------------------------|--------|
| blocks                      | 8      |
| size (MB)                   | 0.0625 |
| used_blocks                 | 8      |
| size (MB)                   | 0.0625 |
| unused_blocks               | 0      |
| size (MB)                   | 0      |
| expired_blocks              | 7      |
| unexpired_blocks           | 8      |
| unformatted_blocks          | 0      |
| fs1_blocks (0 to 25% free space) | 0 |
| fs2_blocks (25 to 50% free space) | 2 |
| fs3_blocks (50 to 75% free space) | 0 |
| fs4_blocks (75 to 100% free space) | 0 |
| full_blocks                 | 3      |

#### 1.1.4. Order

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 2048  |
| size (MB)                   | 16    |
| used_blocks                 | 2048  |
| size (MB)                   | 16    |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 62    |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 1  |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 24 |
| full_blocks                 | 1921  |

SYS_C00554404

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 1024  |
| size (MB)                   | 8     |
| used_blocks                 | 1024  |
| size (MB)                   | 8     |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 128   |
| unformatted_blocks          | 0     |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 52 |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 0  |
| full_blocks                 | 948   |

#### 1.1.5. ORDERITEM

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 16384 |
| size (MB)                   | 128   |
| used_blocks                 | 16384 |
| size (MB)                   | 128   |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 1024  |
| unformatted_blocks          | 256   |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 1  |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 188 |
| full_blocks                 | 15772 |

SYS_C00554411

| Attribute                   | Value |
|-----------------------------|-------|
| blocks                      | 22528 |
| size (MB)                   | 176   |
| used_blocks                 | 22528 |
| size (MB)                   | 176   |
| unused_blocks               | 0     |
| size (MB)                   | 0     |
| expired_blocks              | 7     |
| unexpired_blocks           | 1024  |
| unformatted_blocks          | 0     |
| fs1_blocks (0 to 25% free space) | 0  |
| fs2_blocks (25 to 50% free space) | 31 |
| fs3_blocks (50 to 75% free space) | 0  |
| fs4_blocks (75 to 100% free space) | 0  |
| full_blocks                 | 22306 |

### 1.2. T-SQL

| TABLE\_NAME | INDEX\_NAME |
| :--- | :--- |
| Customer | PK\_\_Customer\_\_D058768771FC3B31 |
| Product | PK\_\_Product\_\_5EEC79D066B0248C |
| Store | PK\_\_Store\_\_A4B61B11C32E1272 |
| Order | PK\_\_Order\_\_C8AAF6FEE0BB95E1 |
| OrderItem | PK\_\_OrderIte\_\_CD443163FF1856EC |

#### 1.2.1. Customer

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Customer | 300000 | 1753 | 13.7 | 1751 | 13.7 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_Customer\_\_D058768771FC3B31 | 300000 | 673 | 5.3 | 673 | 5.3 |

#### 1.2.2. Product

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Product | 100000 | 529 | 4.1 | 526 | 4.1 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_Product\_\_5EEC79D066B0248C | 100000 | 225 | 1.8 | 225 | 1.8 |

#### 1.2.3. Store

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Store | 1000 | 9 | 0.1 | 6 | 0 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_Store\_\_A4B61B11C32E1272 | 1000 | 9 | 0.1 | 5 | 0 |

#### 1.2.4. Order

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Order | 501057 | 1569 | 12.3 | 1562 | 12.2 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_Order\_\_C8AAF6FEE0BB95E1 | 501057 | 1121 | 8.8 | 1121 | 8.8 |

#### 1.2.5. OrderItem

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OrderItem | 5000000 | 18801 | 146.9 | 18798 | 146.9 |

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PK\_\_OrderIte\_\_CD443163FF1856EC | 5000000 | 23713 | 185.3 | 23705 | 185.2 |

## 2. B+strom

Proč není využití stránek u indexu pro primární klíč tabulky `OrderItem` 100%, jako je tomu u ostatních automaticky vytvořených indexů?

- Nejedná se o "jednoduchý" automaticky inkrementovaný PK, a tedy DBS nemůže použít stejnou optimalizaci využití stránek jako u automaticky inkrementovaného PK.

Jaký je maximální počet klíčů v B-stromu řádu $C = 600$ pro $h = 1$ a
$h = 2$?

- Buď $N$ počet klíčů. Pak $N=C^{h+1}−1$.
- $h=1$: $N=359999$
- $h=2$: $N=215999999$

### 2.1. PL/SQL

|ROWID| = 10 B

| Table    | Index         | Height ($h$) | Blocks | Leaf Pages | Inner Pages | Leaf Items | Inner Items | Pct Used | IO cost bodového dotazu | Maximální IO cost rozsahového dotazu | Odhad počtu listových stránek B-stromu|
|----------|---------------|--------|--------|------------|-------------|------------|-------------|----------|-------------------------|--------------------------------------|---|
| CUSTOMER | SYS_C00554393 | 1      | 640    | 562        | 1           | 300000     | 561         | 100      | 2                       | 563 | 1025 |
| PRODUCT  | SYS_C00554397 | 1      | 256    | 187        | 1           | 100000     | 186         | 100      | 2                       | 188 | 342 |
| STORE    | SYS_C00554400 | 1      | 8      | 2          | 1           | 1000       | 1           | 58       | 3                       | 9 | 3 |
| Order    | SYS_C00554404 | 2      | 1024   | 938        | 3           | 500595     | 937         | 100      | 3                       | 1026 | 1711 |
| ORDERITEM| SYS_C00554411 | 2      | 22528  | 22237      | 62          | 5000004    | 22236       | 56       | 3                       | 22530 | 17090 |

Odhad počtu listových stránek B-stromu: $\dfrac{|\text{Leaf Items}|}{\dfrac{8192}{|\text{ROWID}| + 4} \cdot 0.5}$

### 2.2. T-SQL

|RID| = 8 B

#### 2.2.1. Customer

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 669 | 300000 | 16 | 99.7 | 0.3 |
| 1 | 2 | 669 | 11 | 53.7 | 0 |
| 2 | 1 | 2 | 11 | 0.3 | 0 |

- $h=2$
- IO cost bodového dotazu $h+1=3$
- Maximální IO cost rozsahového dotazu $h+669=671$
- Využití vnitřních stránek: $\dfrac{2\cdot 53.7 + 1\cdot 0.3}{3}=35.9$

#### 2.2.2. Product

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 223 | 100000 | 16 | 99.7 | 95.96 |
| 1 | 1 | 223 | 11 | 35.8 | 0 |

- $h=1$
- IO cost bodového dotazu $h+1=2$
- Maximální IO cost rozsahového dotazu $h+223=224$
- Využití vnitřních stránek: $35.8$

#### 2.2.3. Store

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 3 | 1000 | 16 | 74.1 | 33.33 |
| 1 | 1 | 3 | 11 | 0.5 | 0 |

- $h=1$
- IO cost bodového dotazu $h+1=2$
- Maximální IO cost rozsahového dotazu $h+3=4$
- Využití vnitřních stránek: $0.5$

#### 2.2.4. Order

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1116 | 501057 | 16 | 99.8 | 0.36 |
| 1 | 3 | 1116 | 11 | 59.7 | 66.67 |
| 2 | 1 | 3 | 11 | 0.5 | 0 |

- $h=2$
- IO cost bodového dotazu $h+1=3$
- Maximální IO cost rozsahového dotazu $h+1116=1118$
- Využití vnitřních stránek: $\dfrac{3\cdot 59.7 + 1\cdot 0.5}{4}=44.9$

#### 2.2.5. OrderItem

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 23593 | 5000000 | 20 | 57.6 | 41.18 |
| 1 | 110 | 23593 | 15 | 45 | 1.82 |
| 2 | 1 | 110 | 15 | 23.1 | 0 |

- $h=2$
- IO cost bodového dotazu $h+1=3$
- Maximální IO cost rozsahového dotazu $h+23593=23595$
- Využití vnitřních stránek: $\dfrac{110\cdot 45 + 1\cdot 23.1}{111}=44.8$

## 3. Vytvoření indexu

Proč je položka listového uzlu u tohoto indexu větší než u indexu vytvořeného pro primární klíč?

Proč je tento index větší než index vytvořený pro primární klíč?

- Atribut `lName` je typu `VARCHAR(30)`, PK je `INT`.

### 3.1. PL/SQL

| SEGMENT\_NAME | BLOCKS | SIZE\_MB |
| :--- | :--- | :--- |
| IDX\_LNAME | 1024 | 8 |
| SYS\_C00554393 | 640 | 5 |

IDX\_LNAME:

| H | BLOCKS | LEAF\_PAGES | INNER\_PAGES | LEAF\_ITEMS | INNER\_ITEMS | PCT\_USED |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | 1024 | 899 | 4 | 300000 | 898 | 90 |

### 3.2. T-SQL

| IndexName | ItemCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| idx\_lName | 300000 | 1202 | 9.4 | 981 | 7.7 |
| PK\_\_Customer\_\_D058768771FC3B31 | 300000 | 673 | 5.3 | 673 | 5.3 |

idx\_lName:

| level | page\_count | record\_count | avg\_record\_size | page\_utilization | avg\_frag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 936 | 300000 | 22.958 | 98.8 | 2.56 |
| 1 | 18 | 936 | 26.072 | 18 | 83.33 |
| 2 | 1 | 18 | 26.222 | 6.3 | 0 |

<!-- 
#### Customer

#### Product

#### Store

#### Order

#### OrderItem 
-->