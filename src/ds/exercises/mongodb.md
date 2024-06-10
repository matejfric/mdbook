# MongoDB

- [1. Import dokumentů do kolekce moviedb](#1-import-dokumentů-do-kolekce-moviedb)
- [2. Vytvoření a vložení dokumentu](#2-vytvoření-a-vložení-dokumentu)
- [3. Dotazování kolekce](#3-dotazování-kolekce)
- [4. Úkoly](#4-úkoly)

## 1. Import dokumentů do kolekce moviedb

```shell
mongosh.exe mongodb://bac027:bac027@dbsys.cs.vsb.cz:27017
use bac027

db.moviedb.find().count();

for %i in (*.json) DO mongoimport.exe /db:bac027 /collection:fri0089 /authenticationDatabase:admin /username:bac027 /password:bac027 --legacy mongodb://dbsys.cs.vsb.cz:27017 --file %i

db.fri0089.find().count();
```

Počet dokumentů: **98**.

## 2. Vytvoření a vložení dokumentu

| \_id | actor | director | genre | music | name | name\_orig | post | production\_country | rating | year | cinematography | writer |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 661e2ef4033ef960be4f69a9 | \["Hailee Steinfeld", "Kevin Alejandro", "Ella Purnell", "JB Blanc", "Katie Leung", "Harry Lloyd", "Toks Olagundoye", "Jason Spisak", "Shohreh Aghdashloo", "Miles Brown"\] | \["Ash Brannon", "Arnaud Delord", "Pascal Charrue"\] | \["Animovaný", "Fantasy", "Sci-Fi", "Akční", "Dobrodružný", "Drama"\] | \["Alex Seaver", "Alexander Temple"\] | Arcane | Arcane: League of Legends | fri0089 | \["USA"\] | 91 | 2021-2024 | null | null |
| 661e2ef5e1c9e6ec60111905 | \["Timothée Chalamet", "Rebecca Ferguson", "Oscar Isaac", "Jason Momoa", "Josh Brolin", "Javier Bardem", "Stellan Skarsgård", "Zendaya", "Dave Bautista", "David Dastmalchian"\] | Denis Villeneuve | \["Sci-Fi", "Akční", "Dobrodružný", "Drama"\] | Hans Zimmer | Duna | Dune: Part One | fri0089 | \["USA", "Canada"\] | 82 | 2021 | Greig Fraser | Frank Herbert |
| 661e2ef5736ae0b999fdecca | \["Timothée Chalamet", "Zendaya", "Rebecca Ferguson", "Javier Bardem", "Josh Brolin", "Austin Butler", "Florence Pugh", "Dave Bautista", "Christopher Walken", "Léa Seydoux"\] | Denis Villeneuve | \["Sci-Fi", "Akční", "Dobrodružný", "Drama"\] | Hans Zimmer | Duna: Část druhá | Dune: Part Two | fri0089 | \["USA", "Canada"\] | 90 | 2024 | Greig Fraser | Frank Herbert |
| 661e2ef568d9bb00682a9387 | \["Ryan Reynolds", "Jodie Comer", "Joe Keery", "Lil Rel Howery", "Taika Waititi", "Utkarsh Ambudkar", "Channing Tatum", "Britne Oldford", "Camille Kostek", "Mark Lainer"\] | Shawn Levy | \["Sci-Fi", "Akční", "Dobrodružný", "Komedie"\] | Christophe Beck | Free Guy: Hra na hrdinu | Free Guy | fri0089 | \["USA"\] | 74 | 2021 | George Richmond | null |

## 3. Dotazování kolekce

- Dokument s daným jménem.

```sql
db.fri0089.find({ name: "Ztratili jsme Stalina" });
```

- Dokument s daným ObjectId.

```sql
db.fri0089.find({ _id: ObjectId("661e27365d0d1fb83eb71922") });
```

- Dokumenty, jejichž autorem jste vy.

```sql
db.fri0089.find({post: "fri0089"})
```

- Filmy natočené v daném roce a v daném rozsahu roků.

```sql
db.fri0089.find( { year: 2020 } );

db.fri0089.find( { year: {$gte: 2020, $lte: 2024 } } );
```

- Filmy natočené konkrétním režisérem.

```sql
db.fri0089.find( { director: "Christopher Nolan" });
```

- Filmy, ve kterých hraje konkrétní herec a herci.

```sql
db.fri0089.find( { actor: "Timothée Chalamet" });

db.fri0089.find( { actor: {$all: [ "Timothée Chalamet", "Zendaya" ] } } );
```

- Filmy, ve kterých je producentem konkrétní stát, např. CZE.

```sql
db.fri0089.find( { production_country: { $regex: "CZE", $options: "i" } });
```

- Filmy, ke kterým napsal hudbu konkrétní skladatel.

```sql
db.fri0089.find( { music: "Hans Zimmer" });
```

- Filmy s hodnocením větším než 90%.

```sql
db.fri0089.find( { ranking: {$gt: 90} } );
```

- Všechny Sci-Fi filmy.

```sql
db.fri0089.find({ genre: { $regex: "sci-fi", $options: "i" }});
```

- Dotaz s projekcí:

```sql
db.fri0089.find(
  { year: {$gte: 2020, $lte: 2024 } },
  { name: 1, director: 1, _id: 0 }
);
```

- Dotaz s `exists` a projekcí na jméno a režiséra:

```sql
db.fri0089.find(
    { ranking: {$exists: true} },
    { name: 1, director: 1, _id: 0 }
).count()
```

## 4. Úkoly

```js
db.fri0089.deleteMany({}); // delete all documents
// for %i in (*.json) DO mongoimport.exe /db:bac027 /collection:fri0089 /authenticationDatabase:admin /username:bac027 /password:bac027 --legacy mongodb://dbsys.cs.vsb.cz:27017 --file %i

// 10.1, D1
db.fri0089.find().count();
//218

// 10.2, D1
db.fri0089.distinct("name").length; 
// 157

// 10.2, D2
db.fri0089.distinct("movie").length;
// 23

// 10.2, D3
db.fri0089.find({$or: [{ "name": { $exists: true } }, { "movie": { $exists: true } }]}, {name:1, movie:1, _id:0});
// | movie | name |
// | :--- | :--- |
// | Lost update | null |
// | Misery | null |
// | Osviceni | null |
// | null | Forrest Gump |
// | null | Forrest Gump |

// 10.2, D4
db.fri0089.find({$or: [{ "name": { $exists: true } }, { "movie": { $exists: true } }]}).count(); // NE
// 218

// 10.2, A1
db.fri0089.updateMany({ "movie": { $exists: true } }, {$rename: { "movie":"name" }});

// 10.2, D5
db.fri0089.find({ "name": { $exists: true } }).count(); 
// 218

// 10.2, D6
db.fri0089.distinct("name").length;
// 179

// 10.2, D7
db.fri0089.find({ "name": { $exists: true } }).count() - db.fri0089.distinct("name").length;
// 39

// 10.3, D1
db.fri0089.aggregate([
  {
    $group: {
      _id: "$name",
      count: { $sum: 1 },
    },
  },
  {
    $match: {
      count: { $gt: 1 },
    },
  },
  {
    $project: {
      _id: 0,
      name: "$_id",
      count: 1,
    },
  },
]);
// | count | name |
// | :--- | :--- |
// | 2 | Interstellar |
// | 2 | Avatar |
// | 3 | Ztratili jsme Stalina |
// | 2 | Avengers: Endgame |
// | 6 | Vykoupení z věznice Shawshank |


// 10.4, M1
// Find all duplicate films by name and their count
var duplicates = db.fri0089.aggregate([
  {
    $group: {
      _id: "$name",
      count: { $sum: 1 },
    },
  },
  {
    $match: {
      count: { $gt: 1 },
    },
  },
  {
    $project: {
      _id: 0,
      name: "$_id",
      count: 1,
    },
  },
]).toArray();
print(duplicates)
// Loop through each duplicate film name
duplicates.forEach(function (duplicate) {
    var largestDoc = null;
    var largestDocSize = 0;

    var docs = db.fri0089.find({ name: duplicate.name }).toArray();

    // Find the largest document
    docs.forEach(function (doc) {
        var docSize = bsonsize(doc);
        if (docSize > largestDocSize) {
            largestDoc = doc;
            largestDocSize = docSize;
        }
    });

    // Delete all documents except the largest one
    docs.forEach(function (doc) {
        if (doc._id !== largestDoc._id) {
            db.fri0089.deleteOne({ _id: doc._id });
        }
    });

    print("Kept: " + duplicate.name + " (" + largestDocSize + " bytes)");
});

// 10.4, D1
db.fri0089.find().count();
// 179

// 10.4, D2
db.fri0089.distinct("name").length;
// 179

// 10.5, D1
db.fri0089.find({ "genre": { $exists: true } }).count(); 
// 94

// 10.5, D2
db.fri0089.find({ "type": { $exists: true } }).count();
// 19

// 10.5, D3
db.fri0089.find({$and: [{ "type": { $exists: false } }, { "genre": { $exists: false } }]}).count(); 
// 66

// 10.5, A1
db.fri0089.updateMany({ "type": { $exists: true } }, {$rename: { "type":"genre" }});

// 10.5, D4
db.fri0089.find({ "genre": { $exists: true } }).count();
// 113

// 10.5, D5
db.fri0089.find({ "genre": { $exists: false } }).count();
// 66

// 10.6, D1
db.fri0089.distinct("genre"); 
// | result |
// | :--- |
// | Action |
// | Adventure |
// | Akcni |
// | Akční |
// | Ak�n� |


// 10.7, D1
historical_genres = ["Historický", "Historic", "Historical", "history"];
db.fri0089.find({
    "genre": {"$in": historical_genres}
}).count() 
// 10

// 10.7, A1-3
db.fri0089.updateMany({
    "genre": {"$in": historical_genres}
},
{
    $set: { "genre.$": "Historický" }
});

// 10.7, D2
db.fri0089.find({
    genre: "Historický"
}).count()
// 10

// 10.8, D1
db.fri0089.distinct(
    "production_country"
)
// | result |
// | :--- |
// | BGD |
// | BHR |
// | BHS |
// | BLR |
// | CA |

// 10.9, D1
prod_country_cz = ["CZ", "CZE", "Czech Republic"];
db.fri0089.find({
    "production_country": {"$in": prod_country_cz}
}).count()
// 6

// 10.9, A
db.fri0089.updateMany({
    "production_country": {"$in": prod_country_cz}
},
{
    $set: { "production_country.$": "CZE" }
});

// 10.9, D2
db.fri0089.find({"production_country": "CZE"}).count()
// 6
```
