# Heap

## 1. Generate Products

```sql
-- Perform @rec_count inserts into Product table
 -- The values of attributes will be the following:
  -- idProduct will be from 0 to @rec_count 
  -- name will be a concatenation of random name from @gtt_product and number between @product_name_min and @product_name_max
  -- unit_price will be between @gtt_product.min_price and @gtt_product.max_price
  -- producer will be a random value taken from @gtt_producer
  -- decription will be null
 -- Main loop generating the @rec_count Customers

DECLARE @product_min_price int
DECLARE @product_max_price int
DECLARE @product_idx int
DECLARE @product_price int
DECLARE @product_num int

DECLARE @i int = 0
WHILE @i < @rec_count
BEGIN
    SET @product_idx = cast(rand() * @cnt_product as int)
    SET @product_num = cast(rand() * (@product_name_max - @product_name_min) + @product_name_min as INT)
    SET @product_min_price = (SELECT min_price FROM @gtt_product WHERE id = @product_idx)
    SET @product_max_price = (SELECT max_price FROM @gtt_product WHERE id = @product_idx)
    SET @product_price = cast(rand() * (@product_max_price - @product_min_price) + @product_min_price as INT)

    INSERT INTO Product(idProduct, name, unit_price, producer, description)
    VALUES(@i,
            (SELECT name + cast(@product_num as VARCHAR(2)) FROM @gtt_product WHERE id = @product_idx),
            @product_price,
            (SELECT name FROM @gtt_producer WHERE id = cast(rand() * @cnt_producer as int)),
            NULL
        )

    SET @i = @i + 1
END
```

| idProduct | name | unit\_price | producer | description |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Mixér41 | 3907 | Česká zbrojovka | null |
| 1 | HDD4 | 4427 | Apple | null |
| 2 | Vrtací kladivo38 | 20899 | Sram | null |
| 3 | Mikrovlná trouba22 | 8653 | Siemens | null |
| 4 | Diskové pole22 | 871993 | Eta | null |
| 5 | Traktor3 | 2125690 | Eta | null |
| 6 | Server44 | 744929 | John Deere | null |
| 7 | Varná konvice43 | 1580 | LG | null |
| 8 | Svářečka9 | 33587 | Fiat | null |
| 9 | Odpružená vidlice45 | 20685 | Saab | null |
| 10 | HDD2 | 10207 | Specialized | null |
| 11 | SDD43 | 8400 | Specialized | null |
| 12 | Pistol3 | 22737 | Siemens | null |
| 13 | Vrtací kladivo42 | 26513 | Hilti | null |
| 14 | Tepelné čerpadlo2 | 332311 | Samsung | null |
| 15 | Odpružená vidlice35 | 25419 | Škoda | null |
| 16 | Pračka43 | 19901 | Škoda | null |
| 17 | Horské kolo12 | 101063 | Narex | null |
| 18 | Pračka15 | 27593 | Siemens | null |
| 19 | Telefon5 | 16587 | Sram | null |
| 20 | Notebook26 | 31003 | Rheinmetall | null |

## 2. T-SQL

```sql
-- count
select count(*) as order_count
from "Order";

select count(*) as order_item_count
from OrderItem;

select count(*) as customer_count
from Customer;

select count(*) as product_count
from Product;

select count(*) as store_count
from Store;

--pages
PrintPages 'Customer';
PrintPages 'Order';
PrintPages 'OrderItem';
PrintPages 'Product';
PrintPages 'Store';

delete from [OrderItem];
delete from [Order];

--pages after delete
PrintPages 'Order';
PrintPages 'OrderItem';

-- rebuild
ALTER TABLE [OrderItem] REBUILD;
ALTER TABLE [Order] REBUILD;

--pages after rebuild
PrintPages 'Order';
PrintPages 'OrderItem';

EXEC generate_orders;

--pages after insert
PrintPages 'Order';
PrintPages 'OrderItem';
```

### 2.1. Count

|    Table        |  Count |
|:----------------|-------:|
|     Order       | 501,132 |
|   ORDERITEM     | 5,000,000 |
|    CUSTOMER     | 300,000 |
|    PRODUCT      | 100,000 |
|     STORE       | 1,000 |

### 2.2. Pages

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Customer | 300000 | 1753 | 13.7 | 1751 | 13.7 |
| Order | 501132 | 1569 | 12.3 | 1563 | 12.2 |
| OrderItem | 5000000 | 6249 | 48.8 | 6248 | 48.8 |
| Product | 100000 | 529 | 4.1 | 526 | 4.1 |
| Store | 1000 | 9 | 0.1 | 6 | 0 |

### 2.3. Pages After Delete

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Order | 0 | 1569 | 12.3 | 1563 | 12.2 |
| OrderItem | 0 | 6249 | 48.8 | 6248 | 48.8 |

### 2.4. Pages After Rebuild

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Order | 0 | 9 | 0.1 | 2 | 0 |
| OrderItem | 0 | 9 | 0.1 | 2 | 0 |

### 2.5. Pages After Insert

| TableName | RowCounts | TotalPages | TotalPages\_MB | UsedPages | UsedPages\_MB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Order | 501057 | 1569 | 12.3 | 1562 | 12.2 |
| OrderItem | 5000000 | 18801 | 146.9 | 18798 | 146.9 |

## 3. PL/SQL

```sql
select count(*)
from "Order";

select count(*)
from ORDERITEM;

select count(*)
from CUSTOMER;

select count(*)
from PRODUCT;

select count(*)
from STORE;

BEGIN
    DBMS_OUTPUT.PUT_LINE('Order');
    PrintPages('Order', 'FRI0089');

    DBMS_OUTPUT.PUT_LINE('ORDERITEM');
    PrintPages('ORDERITEM', 'FRI0089');

    DBMS_OUTPUT.PUT_LINE('CUSTOMER');
    PrintPages('CUSTOMER', 'FRI0089');

    DBMS_OUTPUT.PUT_LINE('PRODUCT');
    PrintPages('PRODUCT', 'FRI0089');

    DBMS_OUTPUT.PUT_LINE('STORE');
    PrintPages('STORE', 'FRI0089');
END;

delete from ORDERITEM;
delete from "Order";

BEGIN
    PrintPages('Order', 'FRI0089');
    PrintPages('ORDERITEM', 'FRI0089');
END;

ALTER TABLE ORDERITEM ENABLE ROW MOVEMENT;
ALTER TABLE ORDERITEM SHRINK SPACE;

ALTER TABLE "Order" ENABLE ROW MOVEMENT;
ALTER TABLE "Order" SHRINK SPACE;

BEGIN
    PrintPages('Order', 'FRI0089');
    PrintPages('ORDERITEM', 'FRI0089');
END;

BEGIN
    GENERATE_ORDERS();
END;

BEGIN
    PrintPages('Order', 'FRI0089');
    PrintPages('ORDERITEM', 'FRI0089');
END;
```

### 3.1. Count

|    Table        |  Count |
|:----------------|-------:|
|     Order       | 499,390 |
|   ORDERITEM     | 5,000,005 |
|    CUSTOMER     | 300,000 |
|    PRODUCT      | 100,000 |
|     STORE       | 1,000 |

### 3.2. Pages

```text
--Order
blocks: 2048
size (MB): 16
used blocks: 2048
size used (MB): 16
unused blocks: 0
size unused (MB): 0

--ORDERITEM
blocks: 16128
size (MB): 126
used blocks: 16128
size used (MB): 126
unused blocks: 0
size unused (MB): 0

--CUSTOMER
blocks: 2048
size (MB): 16
used blocks: 2048
size used (MB): 16
unused blocks: 0
size unused (MB): 0

--PRODUCT
blocks: 640
size (MB): 5
used blocks: 640
size used (MB): 5
unused blocks: 0
size unused (MB): 0

--STORE
blocks: 8
size (MB): ,0625
used blocks: 8
size used (MB): ,0625
unused blocks: 0
size unused (MB): 0
```

### 3.3. After DELETE

```text
--Order
blocks: 2048
size (MB): 16
used blocks: 2048
size used (MB): 16
unused blocks: 0
size unused (MB): 0

--OrderItem
blocks: 16128
size (MB): 126
used blocks: 16128
size used (MB): 126
unused blocks: 0
size unused (MB): 0
```

### 3.4. After SHRINK SPACE

```text
--Order
blocks: 8
size (MB): ,0625
used blocks: 4
size used (MB): ,03125
unused blocks: 4
size unused (MB): ,03125

--OrderItem
blocks: 8
size (MB): ,0625
used blocks: 4
size used (MB): ,03125
unused blocks: 4
size unused (MB): ,03125
```

### 3.5. After Insert

```text
--Order
blocks: 2048
size (MB): 16
used blocks: 2048
size used (MB): 16
unused blocks: 0
size unused (MB): 0

--OrderItem
blocks: 16384
size (MB): 128
used blocks: 16384
size used (MB): 128
unused blocks: 0
size unused (MB): 0
```
