# AZD

- [1. Druhy dat, předzpracování dat, vlastnosti dat. Výběr atributů (zdůvodnění, princip, entriopie, Gini index, …)](#1-druhy-dat-předzpracování-dat-vlastnosti-dat-výběr-atributů-zdůvodnění-princip-entriopie-gini-index-)
- [2. Hledání častých vzorů v datech (základní principy, metody, varianty, implementace)](#2-hledání-častých-vzorů-v-datech-základní-principy-metody-varianty-implementace)
- [3. Shlukovací metody (shlukování pomocí reprezentantů, hierarchické shlukování). Shlukování na základě hustoty, validace shluků, pokročilé metody shlukování (CLARANS, BIRCH, CURE)](#3-shlukovací-metody-shlukování-pomocí-reprezentantů-hierarchické-shlukování-shlukování-na-základě-hustoty-validace-shluků-pokročilé-metody-shlukování-clarans-birch-cure)
- [4. Rozhodovací stromy (princip, algoritmus, metriky pro vhodnou volbu hodnot dělících atributů, prořezávání)](#4-rozhodovací-stromy-princip-algoritmus-metriky-pro-vhodnou-volbu-hodnot-dělících-atributů-prořezávání)
- [5. Pravděpodobnostní klasifikace (Bayesovský teorém, naivní Bayesovský teorém)](#5-pravděpodobnostní-klasifikace-bayesovský-teorém-naivní-bayesovský-teorém)
- [6. Support Vector Machines (princip, algoritmus, kernel trick)](#6-support-vector-machines-princip-algoritmus-kernel-trick)
- [7. Neuronové sítě (základní princip, metody učení, aktivační funkce)](#7-neuronové-sítě-základní-princip-metody-učení-aktivační-funkce)
- [8. Vyhodnocení klasifikačních algoritmů (chybovost, přesnost, pokrytí, f-metrika)](#8-vyhodnocení-klasifikačních-algoritmů-chybovost-přesnost-pokrytí-f-metrika)
- [9. Regrese (lineární a nelineární regrese, regresní stromy, metody vyhodnocení kvality modelu)](#9-regrese-lineární-a-nelineární-regrese-regresní-stromy-metody-vyhodnocení-kvality-modelu)
- [10. Typy sítí. Graf a matice sousednosti jako reprezentace sítě. Datové struktury pro reprezentaci různých typů sítí, výhody a nevýhody (matice sousednosti, seznamy sousedů, stromy sousedů), složitost operací, hybridní reprezentace](#10-typy-sítí-graf-a-matice-sousednosti-jako-reprezentace-sítě-datové-struktury-pro-reprezentaci-různých-typů-sítí-výhody-a-nevýhody-matice-sousednosti-seznamy-sousedů-stromy-sousedů-složitost-operací-hybridní-reprezentace)
- [11. Topologické vlastnosti sítí, charakteristické hodnoty a jejich distribuce (stupeň, délka cesty, průměr, shlukovací koeficient), typy centralit](#11-topologické-vlastnosti-sítí-charakteristické-hodnoty-a-jejich-distribuce-stupeň-délka-cesty-průměr-shlukovací-koeficient-typy-centralit)
- [12. Globální vlastnosti sítí (malý svět, bezškálovost, růst a preferenční připojování). Mocninný zákon a jeho interpretace v prostředí reálných sítí. Assortarivita](#12-globální-vlastnosti-sítí-malý-svět-bezškálovost-růst-a-preferenční-připojování-mocninný-zákon-a-jeho-interpretace-v-prostředí-reálných-sítí-assortarivita)
- [13. Modely sítí a jejich vlastnosti (Erdös–Rényi, Watts–Strogatz, Barabási–Albert)](#13-modely-sítí-a-jejich-vlastnosti-erdösrényi-wattsstrogatz-barabásialbert)
- [14. Komunity. Globální a lokální přístupy. Modularita](#14-komunity-globální-a-lokální-přístupy-modularita)
- [15. Jiné (pokročilé) modely sítí - modely orientované na komunitní strukturu, temporální sítě](#15-jiné-pokročilé-modely-sítí---modely-orientované-na-komunitní-strukturu-temporální-sítě)
  - [15.1. Temporální sítě](#151-temporální-sítě)
- [16. Odolnost sítí, šíření jevů v sítích. Šíření a maximalizace vlivu v sítích. Predikce linků. Sampling](#16-odolnost-sítí-šíření-jevů-v-sítích-šíření-a-maximalizace-vlivu-v-sítích-predikce-linků-sampling)
  - [16.1. Šíření a maximalizace vlivu v sítích](#161-šíření-a-maximalizace-vlivu-v-sítích)
  - [16.2. Predikce linků](#162-predikce-linků)
- [17. Vícevrstvé sítě, jejich typy a reprezentace. Metody analýzy a vizualizace vícevrstvých sítí, projekce, zploštění](#17-vícevrstvé-sítě-jejich-typy-a-reprezentace-metody-analýzy-a-vizualizace-vícevrstvých-sítí-projekce-zploštění)
  - [17.1. Zploštění](#171-zploštění)
  - [17.2. Projekce](#172-projekce)
- [18. Lokální a globální vlastnosti vícevrstvých sítí, typy centralit a náhodné procházky. Metody detekce komunit ve vícevrstvých sítích](#18-lokální-a-globální-vlastnosti-vícevrstvých-sítí-typy-centralit-a-náhodné-procházky-metody-detekce-komunit-ve-vícevrstvých-sítích)
- [19. Algoritmy pro pattern matching (Vyhledávání jednoho vzorku, více vzorků; Vyhledávání regulárních výrazů; Přibližné vyhledávání)](#19-algoritmy-pro-pattern-matching-vyhledávání-jednoho-vzorku-více-vzorků-vyhledávání-regulárních-výrazů-přibližné-vyhledávání)
- [20. Dokumentografické informační systémy (DIS) (modely DIS - booleovský, vektorový, lexikální analýza, stemming a lematizace, stop slova, konstrukce indexů, vyhodnocení dotazu, relevance, přesnost, úplnost, F-míra)](#20-dokumentografické-informační-systémy-dis-modely-dis---booleovský-vektorový-lexikální-analýza-stemming-a-lematizace-stop-slova-konstrukce-indexů-vyhodnocení-dotazu-relevance-přesnost-úplnost-f-míra)
- [21. Lineární algebra v DIS (metody redukce dimenze, rozklady matic, latentní sémantika, analýza hypertextových dokumentů, PageRank)](#21-lineární-algebra-v-dis-metody-redukce-dimenze-rozklady-matic-latentní-sémantika-analýza-hypertextových-dokumentů-pagerank)
- [22. Neuronové sítě a zpracování textu (word embedding, klasifikace textu, generování textu, …)](#22-neuronové-sítě-a-zpracování-textu-word-embedding-klasifikace-textu-generování-textu-)
- [23. Popište architekturu konvolučních neuronových sítí, použité vrstvy, princip fungování, základní typy architektur](#23-popište-architekturu-konvolučních-neuronových-sítí-použité-vrstvy-princip-fungování-základní-typy-architektur)
- [24. Popište architekturu rekurentních neuronových sítí, typy neuronů, princip fungování](#24-popište-architekturu-rekurentních-neuronových-sítí-typy-neuronů-princip-fungování)

## 1. Druhy dat, předzpracování dat, vlastnosti dat. Výběr atributů (zdůvodnění, princip, entriopie, Gini index, …)

## 2. Hledání častých vzorů v datech (základní principy, metody, varianty, implementace)

## 3. Shlukovací metody (shlukování pomocí reprezentantů, hierarchické shlukování). Shlukování na základě hustoty, validace shluků, pokročilé metody shlukování (CLARANS, BIRCH, CURE)

## 4. Rozhodovací stromy (princip, algoritmus, metriky pro vhodnou volbu hodnot dělících atributů, prořezávání)

## 5. Pravděpodobnostní klasifikace (Bayesovský teorém, naivní Bayesovský teorém)

## 6. Support Vector Machines (princip, algoritmus, kernel trick)

## 7. Neuronové sítě (základní princip, metody učení, aktivační funkce)

## 8. Vyhodnocení klasifikačních algoritmů (chybovost, přesnost, pokrytí, f-metrika)

## 9. Regrese (lineární a nelineární regrese, regresní stromy, metody vyhodnocení kvality modelu)

## 10. Typy sítí. Graf a matice sousednosti jako reprezentace sítě. Datové struktury pro reprezentaci různých typů sítí, výhody a nevýhody (matice sousednosti, seznamy sousedů, stromy sousedů), složitost operací, hybridní reprezentace

- Sociální sítě
- Biologické sítě (interakce proteinů)
- Komunikační sítě (např. e-maily)
- Sítě spolupráce/citační sítě
- Dopravní sítě

## 11. Topologické vlastnosti sítí, charakteristické hodnoty a jejich distribuce (stupeň, délka cesty, průměr, shlukovací koeficient), typy centralit

## 12. Globální vlastnosti sítí (malý svět, bezškálovost, růst a preferenční připojování). Mocninný zákon a jeho interpretace v prostředí reálných sítí. Assortarivita

## 13. Modely sítí a jejich vlastnosti (Erdös–Rényi, Watts–Strogatz, Barabási–Albert)

## 14. Komunity. Globální a lokální přístupy. Modularita

## 15. Jiné (pokročilé) modely sítí - modely orientované na komunitní strukturu, temporální sítě

### 15.1. Temporální sítě

Hrany mohou existovat jen v určitou chvíli (přerušovaně, např. e-maily). Hrany mohou mít měnící se váhu. Obvykle musíme zvolit nějaké časové období (třeba měsíc/rok) a analyzujeme síť vytvořenou z hran, které vznikly v tomto období. Může nás také zajímat vývoj v čase, případně vývoj strukturálních vlastností sítě v čase.

## 16. Odolnost sítí, šíření jevů v sítích. Šíření a maximalizace vlivu v sítích. Predikce linků. Sampling

### 16.1. Šíření a maximalizace vlivu v sítích

Využití např. pro marketing a epidemiologii. Cílem je zjistit, jak se šíří informace v síti hledáním vrcholů s velkým vlivem.

Šíření vlivu/informace lze simulovat např. pomocí modelů *susceptible-infected* ([SI](https://matejfric.github.io/mdbook/ns/ns.html#131-susceptible-infected-si-model)) a independent-cascade ([IC](https://matejfric.github.io/mdbook/ns/ns.html#132-nez%C3%A1visl%C3%BD-kask%C3%A1dov%C3%BD-model-%C5%A1%C3%AD%C5%99en%C3%AD-independent-cascade-model)). Při maximalizaci vlivu se snažíme zvolit minimální počet aktérů a pokrýt maximální (relevantní) část sítě. V sociálních sítích obvykle stačí 3-4 vlivní aktéři k dostatečnému pokrytí (cena/výkon).

### 16.2. Predikce linků

Predikce, že vznikne nová hrana nebo hrana zanikne.

Zajímá nás, které vrcholy by měly být v čase $t+1$ nově spojené a které hrany by měly zaniknout.

Metody založené na podobnosti:

- [Common neighbors](https://matejfric.github.io/mdbook/ns/ns.html#1041-common-neighbors-cn)
- [Jaccard Coefficient](https://matejfric.github.io/mdbook/ns/ns.html#1042-jaccard-coefficient-jc)
- [Preferential attachment](https://matejfric.github.io/mdbook/ns/ns.html#1044-preferential-attachment-pa)
- [Adamic Adar](https://matejfric.github.io/mdbook/ns/ns.html#1043-adamic-adar-index-aa)
- [Resource Allocation index](https://matejfric.github.io/mdbook/ns/ns.html#1045-resource-allocation-index-ra)
- [Cosine similary (Salton index)](https://matejfric.github.io/mdbook/ns/ns.html#1046-cosine-similarity-or-salton-index)
- [Sorensen index](https://matejfric.github.io/mdbook/ns/ns.html#1047-sorensen-index)

[Vyhodnocení predikce linků](https://matejfric.github.io/mdbook/ns/ns.html#105-vyhodnocen%C3%AD-predikce-link%C5%AF):

- Precision, Recall, F1-score, Accuracy, ...

## 17. Vícevrstvé sítě, jejich typy a reprezentace. Metody analýzy a vizualizace vícevrstvých sítí, projekce, zploštění

Vícevrstvá síť je čtveřice (A, $\mathcal{L}$, V, E), kde $G=(V,E)$ je graf, $A\subseteq V\times \mathcal{L}$ je množina aktérů a $\mathcal{L}$ je množina vrstev.

Například můžeme mít třívrstvou homogenní sociální síť, kde někteří lidé (tj. aktéři) spolu pracují, chodí na oběd a chodí běhat.

### 17.1. Zploštění

1. Nevážené.
2. Vážené.

### 17.2. Projekce

Z heterogenní sítě na homogenní.

## 18. Lokální a globální vlastnosti vícevrstvých sítí, typy centralit a náhodné procházky. Metody detekce komunit ve vícevrstvých sítích

- Stupeň aktéra *(degree centrality)* $a$ je suma stupňů vrcholů příslušných $a$ ve vrstvách $L\subseteq{\mathcal{L}}$.
- Sousedé *(neigbors)* aktéra $a$ jsou unikátní sousedé vrcholů příslušných $a$ ve vrstvách $L\subseteq{\mathcal{L}}$.
- Sousedství *(neighborhood)* je počet sousedů aktéra.
- Exluzivní sousedství *(exclusive neighborhood)* je počet sousedů bez sousedů z vrstev $\mathcal{L}\setminus L$.
- Relevance vrcholu je centralita sousedství děleno počtem sousedů ve všech vrstvách (tzn. sousedství pro $\mathcal{L}$).
- Exkluzivní relevance vrcholu je centralita exkluzivního sousedství děleno počtem všech sousedů.

Náhodná procházka ve vícevrstvé síti: V každém kroku má náhodný chodec uniformní pravděpodobnost, že se buď přemístí do jednoho ze sousedních vrcholů v rámci vrstvy, ve které se nachází, nebo přejde do svého protějšku v jiné vrstvě (pokud takový protějšek existuje).

Centralita obsazenosti *(occupation centrality)* aktéra $a$ je pravděpodobnost, že se náhodný chodec na vícevrstvé síti nachází na jakémkoliv vrcholu příslušném $a$.

## 19. Algoritmy pro pattern matching (Vyhledávání jednoho vzorku, více vzorků; Vyhledávání regulárních výrazů; Přibližné vyhledávání)

## 20. Dokumentografické informační systémy (DIS) (modely DIS - booleovský, vektorový, lexikální analýza, stemming a lematizace, stop slova, konstrukce indexů, vyhodnocení dotazu, relevance, přesnost, úplnost, F-míra)

## 21. Lineární algebra v DIS (metody redukce dimenze, rozklady matic, latentní sémantika, analýza hypertextových dokumentů, PageRank)

## 22. Neuronové sítě a zpracování textu (word embedding, klasifikace textu, generování textu, …)

## 23. Popište architekturu konvolučních neuronových sítí, použité vrstvy, princip fungování, základní typy architektur

## 24. Popište architekturu rekurentních neuronových sítí, typy neuronů, princip fungování
