# Databázové systémy

- [1. Struktura PL/SQL bloku](#1-struktura-plsql-bloku)
- [2. Transakce](#2-transakce)
- [3. Procedury](#3-procedury)
  - [3.1. Anonymní procedury](#31-anonymní-procedury)
  - [3.2. Pojmenované procedury](#32-pojmenované-procedury)
  - [3.3. Pojmenované funkce](#33-pojmenované-funkce)
- [4. Trigger](#4-trigger)
- [5. Podmínky](#5-podmínky)
- [6. Cykly](#6-cykly)
- [7. Kurzory](#7-kurzory)
- [8. Hromadné operace](#8-hromadné-operace)

PL/SQL je procedurální rozšíření jazyka SQL.

## 1. Struktura PL/SQL bloku

1. `DECLARE` - nepovinná deklarace lokálních proměnných (prefix `v_`),
2. `BEGIN` - povinné otevření bloku příkazů,
3. `EXCEPTION` - nepovinné zachytávání výjimek,
4. `END` - povinné ukončení bloku.

```sql
v_vek := 20
SELECT vek INTO v_vek
    FROM student
    WHERE login LIKE 'bon007'
```

`SELECT` musí vracet *právě jeden záznam*, jinak je vygenerována výjimka `NO_DATA_FOUND` resp. `TOO_MANY_ROWS`.

<details><summary>Příklad: PL/SQL proměnné</summary>

```sql
DECLARE
  v_fname VARCHAR2(20); --není vhodné (typ se může změnit)
  v_lname Student.lname%TYPE; --preferujeme operátor %TYPE
  v_email Student.email%TYPE;
  v_st Student%ROWTYPE; --záznam z tabulky
  v_date DATE := SYSDATE; --aktuální čas
  v_num NUMBER NOT NULL := 1 ;
BEGIN
  SELECT fname, lname INTO v_fname, v_lname
  FROM student
  WHERE login = 'bon007';

  SELECT * INTO v_st 
  FROM student 
  WHERE login = 'kra228';
  
  v_email := v_fname || '.' || v_lname || '@vsb.cz';
  
  UPDATE students
  SET email = v_email
  WHERE login = 'bon007';
END;
```

</details>

Záznam z tabulky můžeme reprezentovat strukturovaným datovým typem pomocí operátoru `%ROWTYPE`. Např.:

```sql
v_st Student%ROWTYPE;
```

## 2. Transakce

> **Transakce** je *atomická* operace. Buď je provedena celá nebo není proveden žádný příkaz transakce.

<details><summary>Příklad: PL/SQL transakce</summary>

```sql
CREATE TABLE Person (
  login CHAR(5) PRIMARY KEY,
  email VARCHAR(20) NOT NULL,
  password VARCHAR(15) NOT NULL,
  fname VARCHAR(15) NOT NULL,
  mname VARCHAR(15),
  lname VARCHAR(15) NOT NULL,
  street VARCHAR(30),
  city VARCHAR(30)
);

CREATE TABLE Role (
  idRole INT PRIMARY KEY,
  role VARCHAR(30) NOT NULL
);

CREATE TABLE PersonRole (
  login CHAR(5) REFERENCES Person,
  idRole INT REFERENCES Role,
  PRIMARY KEY(login, idRole)
);

INSERT INTO Role VALUES(1, 'Author');

--transakce
BEGIN
  INSERT INTO Person VALUES(
    'sob28', 'jan.sobota@vsb.cz', 'heslo',
    'Jan', NULL, 'Sobota', NULL, NULL);
  INSERT INTO PersonRole VALUES('sob28', 1);
  COMMIT; --transakce proběhla v pořádku, potvrzení změny databáze
EXCEPTION
  WHEN OTHERS THEN
    ROLLBACK; --zrušení transakce
END;
```

</details>

Při nastavení `SET AUTOCOMMIT ON` jsou operace `COMMIT` a `ROLLBACK` ignorovány!

<details><summary>Příklad: PL/SQL zachycení výjimky</summary>

```sql
BEGIN
  INSERT INTO Student (login, fname, lname)
  VALUES ('bon007', 'James', 'Bond');
EXCEPTION
  WHEN DUP_VAL_ON_INDEX THEN
    DBMS_OUTPUT.PUT_LINE('Hodnota atributu login musí být unikátní!');
  WHEN OTHERS THEN
    DBMS_OUTPUT.PUT_LINE(DBMS_UTILITY.FORMAT_ERROR_STACK);
END;
```

</details>

<details><summary>Příklad: PL/SQL vlastní výjimky</summary>

```sql
DECLARE
  too_many_records EXCEPTION;
  v_records INT;
BEGIN
  SELECT COUNT(*) INTO v_records FROM student;
  IF v_records > 20 THEN
    RAISE too_many_records; --výjimka bude propagována do nadřazeného kódu
  ELSE
    INSERT INTO student (login, fname, lname)
    VALUES ('bon007', 'James', 'Bond');
  END IF;
END;

```

</details>

## 3. Procedury

### 3.1. Anonymní procedury

- Jedná se o PL/SQL block.

- Anonymní procedury jsou nepojmenované procedury, které nemohou být volány z jiné procedury.

```sql
DECLARE
  v_name VARCHAR2(30) := 'michal.kratky@vsb.cz';
BEGIN
  INSERT INTO Email VALUES (v_name);
END;
```

### 3.2. Pojmenované procedury

- Pojmenované procedury obsahují *hlavičku se jménem a parametry* procedury.

- Takovouto proceduru je možné volat z jiných procedur nebo spouštět příkazem `EXECUTE` (zkráceně `EXEC`).

- Na rozdíl od anonymních procedur jsou pojmenované procedury předkompilovány a uloženy v databázi.

```sql
CREATE [OR REPLACE] PROCEDURE jmeno_procedury
  (jmeno_parametru [mod] datovy_typ, ...)
IS | AS
  definice_lokalnich_promennych
BEGIN
  telo_procedury
END [jmeno_procedury];
```

- Pro parametry se používá prefix `p_`.
- `mod` může být `{IN | OUT | IN OUT}` - vstupní, výstupní nebo vstupně výstupní proměnná.
- Proměnné typu `VARCHAR2` nebo `NUMBER` se uvádějí bez závorek, které by specifikovaly jejich velikost.
- Spuštěním tohoto kódu se spustí *kompilace*. Pokud je úspěšná, uloží se pojmenovaná procedura do databáze.

<details><summary>Příklad: Pojmenovaná procedura </summary>

```sql
CREATE OR REPLACE PROCEDURE 
InsertEmail(p_login VARCHAR2)
AS
  v_email VARCHAR2(60);
BEGIN
  SELECT email INTO v_email
  FROM Student
  WHERE login = p_login;

  INSERT INTO Email VALUES (v_email);
END;

EXECUTE InsertEmail ('jan440');
```

</details>

### 3.3. Pojmenované funkce

- (Pojmenované) funkce oproti procedurám specifikují návratový typ a *musí vracet hodnotu*.

```sql
CREATE [OR REPLACE] FUNCTION jmeno_funkce
  (jmeno_parametru [mod] datovy_typ, ...)
  RETURN navratovy_datovy_typ
IS | AS
  definice_lokalnich_promennych
BEGIN
  telo_funkce
END [jmeno_funkce];
```

<details><summary>Příklad: Pojmenovaná funkce </summary>

```sql
CREATE OR REPLACE FUNCTION 
GetStudentEmail (p_login IN Student.login%TYPE)
RETURN Student.email%TYPE
AS
  v_email Student.email%TYPE;
BEGIN
  SELECT email INTO v_email FROM Student
  WHERE login = p_login;
  
  RETURN v_email;
END GetStudentEmail;

--Volání funce z PL/SQL bloku (annonymní procedura).
SET SERVEROUTPUT ON;
DECLARE
  v_result Student.email%TYPE;
BEGIN
  v_result := GetStudentEmail('sob28');
  DBMS_OUTPUT.PUT_LINE(v_result);
END;

--Alternativně je možné místo funkce použít
--výstupní parametry procedury:
CREATE OR REPLACE PROCEDURE GetStudentEmail (
  p_login IN Student.login%TYPE,
  p_email OUT Student.email%TYPE
)
AS
BEGIN
  SELECT email INTO p_email FROM Student
  WHERE login = p_login;
END GetStudentEmail;

--Volání procedury je podobné jako u funkce.
DECLARE
  v_email Student.email%TYPE;
BEGIN
  GetStudentEmail('kra22', v_email);
  DBMS_OUTPUT.PUT_LINE(v_email);
END;
```

</details>

## 4. Trigger

Trigger je PL/SQL blok, který je spouštěn v závislosti na nějakém
příkazu jako je `INSERT`, `UPDATE` nebo `DELETE`.

```sql
CREATE [OR REPLACE] TRIGGER jmeno_triggeru
  {BEFORE | AFTER | INSTEAD OF}
  {INSERT [OR] | UPDATE [OR] | DELETE}
  [OF jmeno_sloupce]
  ON jmeno_tabulky
  [REFERENCING OLD AS stara_hodnota NEW AS nova_hodnota]
  [FOR EACH ROW [WHEN (podminka)]]
BEGIN
  prikazy
END;
```

- `OF jmeno_sloupce` – trigger se spouští jen při aktualizaci atributu `jmeno_sloupce`.
- `ON jmeno_tabulky` – specifikujeme tabulku na kterou se
trigger váže.
- `[FOR EACH ROW [WHEN (podminka)]]`
  - Implicitně se trigger spouští pouze jednou pro jeden příkaz (který může aktualizovat (mazat, vkládat) více záznamů).
  - Tímto volitelným parametrem specifikujeme, že trigger má být spouštěn pro každý záznam, který je SQL příkazem aktualizován.
- `[REFERENCING OLD AS stara_hodnota NEW AS
nova_hodnota]`
  - Umožňuje pojmenovat pomocí proměnné staré a nové hodnoty záznamu se kterým manipulujeme.
  - Implicitně jsou tyto proměnné pojmenovány jako `:OLD` a `:NEW`.

<details><summary>Příklad: PL/SQL trigger </summary>

Při mazání záznamu z tabulky `Student` budeme mazaný záznam ukládat do tabulky `Hist_stud`.

```sql
CREATE OR REPLACE TRIGGER del_student
  BEFORE DELETE
  ON student
  FOR EACH ROW
BEGIN
  INSERT INTO Hist_stud (login, name, surname)
  VALUES (:OLD.login, :OLD.name, :OLD.surname);
END;
```

</details>

Pokud se pokusíme v triggeru číst nebo modifikovat stejnou tabulku, pro kterou je daný trigger definován, obdržíme chybu *mutating table error* (i.e. `ORA-04091`). Obecně bychom se takovému triggeru měli vyhnout, ale lze to obejít pomocí *složeného triggeru*.

<details><summary>Příklad: PL/SQL compound trigger </summary>

```sql
CREATE OR REPLACE TRIGGER compound_trigger
  FOR UPDATE OF salary ON employees
  COMPOUND TRIGGER
  -- Declaration part (optional)
  BEFORE STATEMENT IS
    -- Code before the statement execution
  BEGIN
    -- Implementation for BEFORE STATEMENT
  END BEFORE STATEMENT;

  BEFORE EACH ROW IS
    -- Code before each row update
  BEGIN
    -- Implementation for BEFORE EACH ROW
  END BEFORE EACH ROW;

  AFTER EACH ROW IS
    -- Code after each row update
  BEGIN
    -- Implementation for AFTER EACH ROW
  END AFTER EACH ROW;

  AFTER STATEMENT IS
    -- Code after the statement execution
  BEGIN
    -- Implementation for AFTER STATEMENT
  END AFTER STATEMENT;
  
END compound_trigger;
```

</details>

## 5. Podmínky

```sql
IF podminka1 THEN
  -- příkazy pro podminku1
ELSIF podminka2 THEN
  -- příkazy pro podminku2
ELSE
  -- příkazy, které se provedou, pokud žádná z předchozích podmínek není splněna
END IF;
```

## 6. Cykly

```sql
LOOP
--příkazy cyklu
[ EXIT; | EXIT WHEN podminka; ]
END LOOP;
```

<details><summary>Příklad: Cyklus s podmínkou na konci (LOOP) </summary>

```sql
DECLARE
  v_i INT := 0;
BEGIN
  LOOP
    DBMS_OUTPUT.PUT_LINE('v_i: ' || v_i);
    EXIT WHEN v_i >= 5;
    v_i := v_i + 1;
  END LOOP;
END;
```

</details>

```sql
WHILE podminka LOOP
--příkazy cyklu
END LOOP;
```

<details><summary> Příklad: Cyklus s podmínkou na začátku (WHILE) </summary>

```sql
DECLARE
  v_i INT := 0;
BEGIN
  WHILE v_i < 6 LOOP
    DBMS_OUTPUT.PUT_LINE('v_i: ' || v_i);
    v_i := v_i + 1;
  END LOOP;
END;
```

</details>

```sql
FOR jmeno_promenne IN [REVERSE] start..end
LOOP
  -- příkazy cyklu
END LOOP;
```

<details><summary> Příklad: Cyklus s pevným počtem opakování (FOR) </summary>

```sql
DECLARE
  v_i INT;
BEGIN
  FOR v_i IN 0..5
  LOOP
    DBMS_OUTPUT.PUT(v_i);
    IF v_i <> 5 THEN
      DBMS_OUTPUT.PUT(', ');
    END IF;
  END LOOP;
  DBMS_OUTPUT.NEW_LINE();
END;
```

</details>

## 7. Kurzory

Kurzory jsou *pomocné proměnné* vytvořené *po* provedení nějakého SQL příkazu.

- **Implicitní kurzor** – vytváří se automaticky po provedení příkazů jako `INSERT`, `DELETE` nebo `UPDATE`.

- **Explicitní kurzor** – definuje se již v definiční části procedury podobně jako proměnná. Takový kurzor je často spojen s příkazem `SELECT`, který vrací více než jeden řádek.

Definice explicitního kurzoru má následující syntaxi:

```sql
CURSOR jmeno_kursoru IS vysledek_prikazu_select;
```

Kde `vysledek_prikazu_select` vrací množinu záznamů. Pomocí kurzoru můžeme postupně *procházet jednotlivé záznamy* výsledku `SELECT`. V každém kroku programu ukazuje kurzor *pouze na jeden záznam* výsledku.

Práce s kurzorem probíhá pomocí následujících příkazů:

```sql
OPEN jmeno_kurzoru
```

- Otevření kurzoru - provedení SQL příkazu spojeného s kurzorem a nastavení kurzoru na první záznam výsledku.

```sql
FETCH jmeno_kurzoru INTO promenna_zaznam
```

- Načtení aktuálního záznamu kurzoru do proměnné `promenna_zaznam` a posunutí se na další záznam.

```sql
CLOSE jmeno_kurzoru
```

- Zavření kurzoru.

<details><summary> Příklad: PL/SQL kurzor pro načtení všech příjmení z tabulky </summary>

```sql
DECLARE
  CURSOR c_surname IS
    SELECT * FROM Student;
    
  v_record Student%ROWTYPE;
  v_tmp INTEGER := 0;
BEGIN
  OPEN c_surname; --open cursor
  LOOP
    FETCH c_surname INTO v_record;
    EXIT WHEN c_surname%NOTFOUND; --%NOTFOUND returns true when
                                  --there is no other record.
    v_tmp := c_surname%ROWCOUNT; --%ROWCOUNT returns the number of records
                                 --obtained by FETCH.
    DBMS_OUTPUT.PUT_LINE(v_tmp || v_record.surname);
  END LOOP;
  CLOSE c_surname; --close cursor
END;


--Simpler solution with FOR LOOP:
DECLARE
  CURSOR c_surname IS
    SELECT surname FROM Student;
    
  v_surname Student.surname%TYPE;
  v_tmp NUMBER := 0;
BEGIN
  FOR one_surname IN c_surname LOOP
    v_tmp := c_surname%ROWCOUNT;
    v_surname := one_surname.surname;
    DBMS_OUTPUT.PUT_LINE(v_tmp || ' ' || v_surname);
  END LOOP;
END;


--Even simpler without an explicit definition of a cursor:
DECLARE
  v_surname Student.surname%TYPE;
  v_tmp NUMBER := 0;
BEGIN
  FOR one_surname IN (SELECT surname FROM Student)
  LOOP
    v_tmp := v_tmp + 1;
    v_surname := one_surname.surname;
    DBMS_OUTPUT.PUT_LINE(v_tmp || ' ' || v_surname);
  END LOOP;
END;

```

</details>

## 8. Hromadné operace

```sql
... BULK COLLECT INTO collection_name[,collection_name] ...
```

- Hromadné zapsání záznamů do kolekce (pole).

```sql
FORALL index IN lower_bound..upper_bound
sql_statement;
```

- Pro každý záznam v kolekci (poli) proveď nějaký SQL příkaz *(není to cyklus!)*.

<details><summary> Příklad: PL/SQL BULK COLLECT </summary>

Tento blok jazyka PL/SQL deklaruje dvě vnořené tabulky `enums` a `names` a poté provede hromadný sběrný dotaz pro získání ID a příjmení zaměstnanců z tabulky `Employees`.

Následně příkazem `FORALL` provede aktualizaci tabulky `Myemp` s odpovídajícími jmény na základě získaných ID zaměstnanců.

```sql
DECLARE
  TYPE NumTab IS TABLE OF Employees.employee_id%TYPE;
  TYPE NameTab IS TABLE OF Employees.last_name%TYPE;
  
  enums NumTab;
  names NameTab;
BEGIN
  SELECT employee_id, last_name
  BULK COLLECT INTO enums, names
  FROM Employees
  WHERE employee_id > 1000;

  --rest of the code...

  FORALL i IN enums.FIRST..enums.LAST
    UPDATE Myemp
    SET name = names(i)
    WHERE Employee = enums(i);
END;
```

</details>

<details><summary> Příklad: PL/SQL vložení 100 000 záznamů </summary>

```sql
DECLARE
  TYPE UserArray IS VARRAY(10000) OF Usertab%ROWTYPE;
  v_userArray UserArray;
  v_counter NUMBER := 0;
  v_start NUMBER DEFAULT DBMS_UTILITY.GET_TIME;
BEGIN
  v_userArray := UserArray(); -- initialization
  v_userArray.EXTEND(10000); -- resize
  
  -- We must run it 10 times because 
  -- 100,000 items must be inserted.
  FOR i IN 1..10 LOOP
    -- Prepare array
    FOR j IN 1..10000 LOOP
      v_counter := v_counter + 1;
      v_userArray(j).id := v_counter;
      v_userArray(j).fname := 'fname' || v_counter;
      v_userArray(j).lname := 'lname' || v_counter;
    END LOOP;
    
    -- Bulk insert with FORALL
    FORALL k IN v_userArray.FIRST..v_userArray.LAST
      INSERT INTO Usertab VALUES v_userArray(k);
  END LOOP;
  
  DBMS_OUTPUT.PUT_LINE(
    ROUND((DBMS_UTILITY.GET_TIME - v_start) / 100, 2) || ' s'
    );
END;
```

Zrychlení za použití hromadného vkládání je v tomto případě asi **8násobné** v porovnání s vkládáním po 1
záznamu.

</details>
