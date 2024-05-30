# Analýza obrazu I

- [1. Segmentace obrazu](#1-segmentace-obrazu)
  - [1.1. Detekce hran s využitím gradientu](#11-detekce-hran-s-využitím-gradientu)
    - [1.1.1. Sobelův filtr](#111-sobelův-filtr)
  - [1.2. Detekce hran hledáním průchodu nulou (též varianta s předchozím rozostřením)](#12-detekce-hran-hledáním-průchodu-nulou-též-varianta-s-předchozím-rozostřením)
  - [1.3. Základní myšlenky Cannyho detekce hran](#13-základní-myšlenky-cannyho-detekce-hran)
    - [1.3.1. Prahování](#131-prahování)
  - [1.4. Prahování](#14-prahování)
  - [1.5. Detekce oblastí](#15-detekce-oblastí)
  - [1.6. Parametrické modely hrany](#16-parametrické-modely-hrany)
  - [1.7. Matematická morfologie](#17-matematická-morfologie)
    - [1.7.1. Dilatace (Dilation)](#171-dilatace-dilation)
    - [1.7.2. Eroze (Errosion)](#172-eroze-errosion)
    - [1.7.3. Otevření (Opening)](#173-otevření-opening)
    - [1.7.4. Uzavření (Closing)](#174-uzavření-closing)
    - [1.7.5. Morfologický gradient](#175-morfologický-gradient)
- [2. Příznakové metody analýzy obrazu](#2-příznakové-metody-analýzy-obrazu)
  - [2.1. Příznaky používané v příznakové analýze obrazů](#21-příznaky-používané-v-příznakové-analýze-obrazů)
    - [2.1.1. Momenty](#211-momenty)
    - [2.1.2. Kruhovost](#212-kruhovost)
    - [2.1.3. Popis tvaru objektu pomocí průběhu křivosti jeho hranice](#213-popis-tvaru-objektu-pomocí-průběhu-křivosti-jeho-hranice)
    - [2.1.4. Příznaky odvozené z histogramu jasu](#214-příznaky-odvozené-z-histogramu-jasu)
  - [2.2. Univerzální příznaky](#22-univerzální-příznaky)
    - [2.2.1. HoG](#221-hog)
    - [2.2.2. Local Binary Patterns (LBP)](#222-local-binary-patterns-lbp)
  - [2.3. Klasifikátor a klasifikace pomocí diskriminačních funkcí](#23-klasifikátor-a-klasifikace-pomocí-diskriminačních-funkcí)
  - [2.4. Klasifikace pomocí etalonů](#24-klasifikace-pomocí-etalonů)
  - [2.5. Stanovení diskriminační funkce metodou minimalizace ztráty](#25-stanovení-diskriminační-funkce-metodou-minimalizace-ztráty)
  - [2.6. Rozpoznání neuronovou sítí (vícevrstvá síť s učením back propagation)](#26-rozpoznání-neuronovou-sítí-vícevrstvá-síť-s-učením-back-propagation)
  - [2.7. Vyhodnocení účinnosti zvolené množiny příznaků](#27-vyhodnocení-účinnosti-zvolené-množiny-příznaků)
  - [2.8. Karhunen-Loéveho transformace](#28-karhunen-loéveho-transformace)
  - [2.9. Rozpoznávání pomocí hlubokých neutonových sítí](#29-rozpoznávání-pomocí-hlubokých-neutonových-sítí)
- [3. Zpětná stereoprojekce](#3-zpětná-stereoprojekce)
  - [3.1. Zpětná stereoprojekce a základní vztahy pro kamery s rovnoběžnými optickými osami](#31-zpětná-stereoprojekce-a-základní-vztahy-pro-kamery-s-rovnoběžnými-optickými-osami)
  - [3.2. Absolutní kalibrace kamery a rekonstrukce](#32-absolutní-kalibrace-kamery-a-rekonstrukce)
  - [3.3. Relativní kalibrace a rekonstrukce](#33-relativní-kalibrace-a-rekonstrukce)
  - [3.4. Principy automatizované rekonstrukce objektů (hledání rohů a korespondencí)](#34-principy-automatizované-rekonstrukce-objektů-hledání-rohů-a-korespondencí)
- [4. Analýza obrazů proměnných v čase](#4-analýza-obrazů-proměnných-v-čase)
  - [4.1. Princip sledování objektů v obrazech](#41-princip-sledování-objektů-v-obrazech)
  - [4.2. Optický tok](#42-optický-tok)
  - [4.3. Základní principy rozpoznávání činností pomocí neuronových sítí](#43-základní-principy-rozpoznávání-činností-pomocí-neuronových-sítí)
- [5. Geometrické transformace obrazu](#5-geometrické-transformace-obrazu)
- [6. Hough transformace](#6-hough-transformace)
- [7. Úlohy při zpracování obrazu](#7-úlohy-při-zpracování-obrazu)
- [8. Template matching](#8-template-matching)
- [9. Typy kernelů](#9-typy-kernelů)
- [10. Detekce objektů a rozpoznávání obličejů](#10-detekce-objektů-a-rozpoznávání-obličejů)
  - [10.1. Sliding window](#101-sliding-window)
  - [10.2. Proces rozpoznávání](#102-proces-rozpoznávání)
  - [10.3. Detekce obrazový zájmový bodů](#103-detekce-obrazový-zájmový-bodů)
  - [10.4. Haar features (příznaky)](#104-haar-features-příznaky)
    - [10.4.1. Kaskádová regrese (Cascaded Regression)](#1041-kaskádová-regrese-cascaded-regression)
    - [10.4.2. Knihovny pro detekci zájmový bodů obličeje](#1042-knihovny-pro-detekci-zájmový-bodů-obličeje)
- [11. Autonomí vozidla](#11-autonomí-vozidla)
  - [11.1. Senzory](#111-senzory)
  - [11.2. Úrovně autonomních vozidel](#112-úrovně-autonomních-vozidel)
- [12. OpenCV](#12-opencv)

## 1. Segmentace obrazu

### 1.1. Detekce hran s využitím gradientu

Hrany mohou být popsány změnami intenzit pixelů. Oblasti s velkými změnami v intenzitách pixelů značí hranu. Můžeme zkoumat změny v intenzitách sousedních pixelů. Pro detekci "něčeho" lze počítat počet *nenulových prvků* ve výsledném obraze detekce hran.

Detekce oblastí stanovením hranice:

<img src="figures/edge-detection.png" alt="edge-detection" width="100px">

Gradientní metody využívají skutečnosti, že v místě hrany má absolutní hodnota první derivace průběhu jasu vysokou hodnotu.

Hranové operátory $\dfrac{\partial f}{\partial x}$ a $\dfrac{\partial f}{\partial y}$.

Pro hrany v libovolném směru lze použít derivaci ve směru.

Velikost hrany vypočteme jako:

$$ \left\lVert \dfrac{\partial f(x,y)}{\partial x}, \dfrac{\partial f(x,y)}{\partial y} \right\rVert_2 $$

<img src="figures/edge-direction.png" alt="edge-direction" width="200px">

Směr gradientu hrany jako:

$$\varphi(x,y)=\arctan\left(\dfrac{f_y(x,y)}{f_x(x,y)}\right)$$

Směr hrany jako:

$$\psi(x,y) = \varphi(x,y) + \frac{\pi}{2}$$

> Při praktické implementaci pracujeme nejčastěji s diskrétními funkcemi $\Rightarrow$ nahradíme derivace konečnými diferencemi:
>
> $$\begin{align*}f_x(x,y)&=f(x + 1,y)-f(x,y)\\f_y(x,y)&=f(x,y+1)-f(x,y)\end{align*}$$

#### 1.1.1. Sobelův filtr

- *horizontální změny*, např. s kernelem $3\times3$ ($\ast$ značí konvoluci):
  - $G_x=\begin{bmatrix}
    -1 & 0 & 1\\
    -2 & 0 & 2\\
    -1 & 0 & 1\\
\end{bmatrix}\ast I$

- *horizontální změny*, např. s kernelem $3\times3$ ($\ast$ značí konvoluci):
  - $G_y=\begin{bmatrix}
    -1 & -2 & 1\\
    0 & 0 & 0\\
    1 & 2 & 1\\
\end{bmatrix}\ast I$

- v každém bodě obrazu lze vypočítat aproximaci gradientu:
  - $G =\sqrt{G_x^2-G_y^2}$  
  - nebo jednodušeji $G =|G_x|+|G_y|$  

Jak funguje konvoluce?

1. Střed kernelu je umístěn na určitý pixel vstupního obrazu.
2. Každý prvek jádra je vynásoben s odpovídajícím pixelem ve vstupním obraze.
3. Sečteme výsledek násobení.
4. Tento výsledek můžeme uložit do nového obrazu (mapy hran).

### 1.2. Detekce hran hledáním průchodu nulou (též varianta s předchozím rozostřením)

Průběh obrazové funkce (jasu) a její první a druhé derivace v místě hrany:

<img src="figures/brightness-and-derivations.png" alt="brightness-and-derivations" width="150px">

Laplacián (pro druhou derivaci):

$$ \dfrac{\partial^2 f(x,y)}{\partial x^2}, \dfrac{\partial^2 f(x,y)}{\partial y^2} $$

### 1.3. Základní myšlenky Cannyho detekce hran

Výsledná funkce vznikla minimalizací funkcionálu, který měří lokalizační chybu, signal2noise ratio (?)

1. Redukce šumu
2. Sobelovy filtry
3. Ztenčování hran - všechny hrany mají tloušťku jednoho pixelu
4. Prahování (thresholding)
  
#### 1.3.1. Prahování

- dva prahy $T_{upper}$ a $T_{lower}$

<img src="figures/canny.webp" alt="canny" width="200px">

- hrana $A$ je "jistá" hrana, protože $A > T_{upper}$
- hrana $B$ je taky hrana, přestože $B < T_{upper}$, protože je spojená s hranou $A$
- $C$ není hrana, protože $C < T_{upper}$ a zároveň $C$ není spojena se "silnou" hranou
- hrana $D$ je automaticky zahozena, protože $D < T_{lower}$

### 1.4. Prahování

- binární prahování
  - $(I_{dst})_{x,y} = \left\{\begin{array}{ll}
        \texttt{maxVal} &\texttt{if}\,\,(I_{src})_{x,y}\,\,>\,\,\texttt{threshold}\\
        0 &\texttt{otherwise}
    \end{array}\right.$

- invertované binární prahování
  - $(I_{dst})_{x,y} = \left\{\begin{array}{ll}
        0 &\texttt{if}\,\,(I_{src})_{x,y}\,\,>\,\,\texttt{threshold}\\
        \texttt{maxVal} &\texttt{otherwise}
    \end{array}\right.$

Uveďte některé metody stanovení prahu (např. minimalizace chyby):

### 1.5. Detekce oblastí

Naznačte princip metody dělení a spojování oblastí:

### 1.6. Parametrické modely hrany

### 1.7. Matematická morfologie

- často používané na binárním obraze
- potřebujeme zdrojový obrázek a **kernel**
- kernel se postupně *přikládá* na zdrojový obrázek a překrývající se pixely se vyhodnocují
- kernel může být třeba `np.ones((5,5),np.uint8)`, nebo taky elipsa, kříž atd.
- lze zobecnit na grayscale popř. barevné obrazy

#### 1.7.1. Dilatace (Dilation)

- pixel bude mít hodnotu **1** právě tehdy, když **alespoň jeden** pixel "pod" kernelem má hodnotu **1**, jinak je erodován (přepsán na nulu)
- zesvětluje barevný obraz

#### 1.7.2. Eroze (Errosion)

- pixel bude mít hodnotu **1** právě tehdy, když **všechny** jeden pixel "pod" kernelem má hodnotu **1**
- ztmavuje barevný obraz

#### 1.7.3. Otevření (Opening)

- eroze následování dilatací
- odstranění šumu

#### 1.7.4. Uzavření (Closing)

- dilatace následování erozí

#### 1.7.5. Morfologický gradient

- rozdíl mezi dilatací a erozí (SAD)

## 2. Příznakové metody analýzy obrazu

### 2.1. Příznaky používané v příznakové analýze obrazů

Několik příkladů příznaků a způsob jejich výpočtu. Příznaky otvozené ze tvaru, tvaru hranice a jasu.

#### 2.1.1. Momenty

Dvourozměrný moment řádu $(p,q)$ pro plochu $\Omega$ je:

$$m_{p,q}=\iint\limits_{\Omega}x^py^qf(x,y)\mathrm{d}x\mathrm{d}y$$

V analýze obrazu máme diskrétní obraz a funkce průběhu jasu je obvykle $f(x,y)=1$:

$$m_{p,q}=\sum\limits_{\Omega}x^py^q$$

Plocha: $m_{0,0}$

Těžiště:

$$
\begin{align*}
    x_t&=\dfrac{m_{1,0}}{m_{0,0}}\\
    y_t&=\dfrac{m_{0,1}}{m_{0,0}}
\end{align*}
$$

Pokud uvažujeme souřadnou soustavu s *osami v těžišti*, tak:

$$\mu_{p,q}=\sum\limits_{\Omega}(x-x_t)^p(y-y_t)^qf(x,y)$$

Momenty $\mu_{p,q}$ k těžištním osám *nezávisí na poloze objektu*, ale *závisí na velikosti a rotaci* objektu.

- &#9645; $\Rightarrow\dfrac{\mu_{2,0}}{\mu_{0,2}}$ větší než 1
- &#9647; $\Rightarrow\dfrac{\mu_{2,0}}{\mu_{0,2}}$ menší než 1

Co kdybychom chtěli, aby tento příznak nebyl závislý na rotaci/orientaci? **Hlavní momenty setrvačnosti**: Představ si natočený obdélník, cílem je najít natočený souřadnicový systém takový, že najdeme minimum a maximum:

$$
\begin{array}{c}
    \mu_{\mathrm{max}} \\
    \mu_{\mathrm{min}}
\end{array}
= \frac{1}{2} \left( \mu_{2,0} + \mu_{0,2} \right) \pm \frac{1}{2} \sqrt{ 4 \mu_{1,1}^{2} + \left( \mu_{2,0} - \mu_{0,2} \right)^2 }.
$$

#### 2.1.2. Kruhovost

Buď $P$ délka hranice objektu a $S$ jeho plocha. Kruhovost definujeme $\boxed{C=\dfrac{P^2}{S}.}$

#### 2.1.3. Popis tvaru objektu pomocí průběhu křivosti jeho hranice

Průběh křivosti:

<img src="figures/curvature.png" alt="curvature" width="350px">

Buď $P$ délka hranice objektu a $k$ funkce křivosti hranice objektu. Informace o křivosti můžeme komprimovat do jedné hodnoty:

$$\dfrac{1}{P}\int\limits_{s}\big(k(s)\big)^2\mathrm{d}s.$$

Pokud je jedno číslo málo, můžeme vzít $n$ prvních členů **Fourierovy řady** (nejčastěji $\approx 5$).

#### 2.1.4. Příznaky odvozené z histogramu jasu

Vhodné pro objekty, které jsou charakteristické svojí texturou nebo jistým rozložením jasu. Buď $b$ jas pixelu, vypočteme histogram pixelů objektu $N(b)$ a provedem normalizaci $p(b)=N(b)/N$.

- střední hodnota,
- rozptyl,
- šikmost,
- křivost
- entropie,
- energie (kontrast).

### 2.2. Univerzální příznaky

#### 2.2.1. HoG

#### 2.2.2. Local Binary Patterns (LBP)

- Ojala et al.
- [Tutorial](https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)
- Hlavní myšlenkou LBP je, že lokální struktury obrazu (mikrovzory, jako jsou čáry, hrany, skvrny a ploché oblasti) lze efektivně zakódovat porovnáním každého pixelu s pixely sousedními.

<img src="figures/lbp.png" alt="lbp" width="500px">
<img src="figures/lbp2.png" alt="lbp2" width="500px">

- z LBP obrazu následně můžeme sestavit histogram atd.

### 2.3. Klasifikátor a klasifikace pomocí diskriminačních funkcí

Klasifikátorem rozumíme zobrazení $f\colon\mathcal{X}\rightarrow\omega$, kde $\mathcal{X}$ je matice příznaků a $\omega$ je vektor tříd, i.e., $\omega=f(\mathbf{x})$.

Klasifikace diskriminačními funkcemi:

<img src="figures/clf-discriminative-funs.png" alt="clf-discriminative-funs" width="400px">

Jednotlivé hustoty odpovídají podmíněné pravděpodobnosti $g_r(\mathbb{x})=\mathcal{P}(\mathbb{x}|\omega_{r})$, kde $r$ je index třídy.

Dále jsme schopni definovat funkce pro každou třídu a minimalizovat plochu pod křivkou (integrál). Dá se ukázat, že tato diskriminační funkce má tvar asi $\mathcal{P}(x|\omega_r)\mathcal{P}(\omega_r)$.

### 2.4. Klasifikace pomocí etalonů

Vysvětlete vztah k diskriminačním funkcím:

### 2.5. Stanovení diskriminační funkce metodou minimalizace ztráty

### 2.6. Rozpoznání neuronovou sítí (vícevrstvá síť s učením back propagation)

<img src="figures/neuron.png" alt="neuron" width="200px">

### 2.7. Vyhodnocení účinnosti zvolené množiny příznaků

### 2.8. Karhunen-Loéveho transformace

### 2.9. Rozpoznávání pomocí hlubokých neutonových sítí

Základní principy výstavby hlubokých neuronových sítí:

## 3. Zpětná stereoprojekce

### 3.1. Zpětná stereoprojekce a základní vztahy pro kamery s rovnoběžnými optickými osami

### 3.2. Absolutní kalibrace kamery a rekonstrukce

### 3.3. Relativní kalibrace a rekonstrukce

### 3.4. Principy automatizované rekonstrukce objektů (hledání rohů a korespondencí)

## 4. Analýza obrazů proměnných v čase

### 4.1. Princip sledování objektů v obrazech

Základní myšlenky Kalmanova filtru:

- Je postavený na fyzikálním modelu, do kterého započítává nejistoty měření.
- Zjednodušený Kalmanův filtr s konstantním zrychlení pro 1D:

$$
\begin{align*}
    \mathbf{H} &= [1,0,0]\\
    \mathbf{z}_t &= \mathbf{H}\mathbf{x}_t+\mathbf{v}_t\\
    \mathbf{x}_{t+1} &= \varPhi_{t\mid t+1} \mathbf{x}_t + \mathbf{w}_t
\end{align*}
$$

$$
\begin{bmatrix}
    s_{t+1}\\
    v_{t+1}\\
    a_{t+1}
\end{bmatrix} =
\begin{bmatrix}
  1 & \Delta t  & \frac{1}{2}\Delta t^2\\
  0 & 1 & \Delta t\\
  0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
    s_{t}\\
    v_{t}\\
    a_{t}
\end{bmatrix} + \mathbf{w}_t
$$

- kde $\mathbf{w}$ je bílý šum (nezávislé NV).

### 4.2. Optický tok

- Analýza obrazů měnících se v čase.

Rovnice optického toku a její řešení: TODO

### 4.3. Základní principy rozpoznávání činností pomocí neuronových sítí

## 5. Geometrické transformace obrazu

- Skripta strana 56.

Afiní transformace, zachovává rovnoběžnost (ortografický pohled):

$$y=Ax+t$$

Perspektivní transformace:

$$y=Px,$$

kde $P\in\mathbb{R^{4,4}}$. Např.

$$y=Px=\begin{bmatrix}
    1 & 0 & 0 & 0\\
    0 & 1 & 0 & 0\\
    0 & 0 & 1 & 0\\
    0 & 0 & 0 & 1\\
\end{bmatrix}\begin{bmatrix}
    x\\
    y\\
    z\\
    1
\end{bmatrix},$$

do matice $P$ je možné "dát" translaci $t$.

Co je epipolára???

## 6. Hough transformace

- [:háfova:]
- detekce bodů na přímce, kružnici (po detekci hran)

## 7. Úlohy při zpracování obrazu

- vylepšení / filtrace
- detekce hran
- detekce objektů
  - nalezení ROI (souřadnice a měřítko objektu)
- rozpoznání objektů
  - identifikace do kategorií
  - rozpoznávání obličejů
- sledování objektů *(object tracking)*
  - odhad trajektorie
- segmentace
  - rozdělení obrazu na více oblastí
- detekce lidské pózy *(human pose detection)*

## 8. Template matching

- jednoduchá metoda pro lokalizaci objektů
- potřebuju *template* a *zdrojový obraz* (source)
- porovnává se template a část zdrojového obraz

Jak porovnat *template* a *source*?

- suma absolutních rozdílů
  - $SAD=\sum\limits_{x,y}\lvert (I_1)_{x,y} - (I_2)_{x,y}\rvert$
- suma čtvercových rozdílů
  - $SSD=\sum\limits_{x,y}\left[ (I_1)_{x,y} - (I_2)_{x,y}\right]^2$
- vzájemná korelace (cross correlation)
  - $CC=\sum\limits_{x,y}\left[ (I_1)_{x,y} \cdot (I_2)_{x,y}\right]$
- odpovídající normované varianty a další

## 9. Typy kernelů

- Sobel $3\times3$
  - $K_x=\begin{bmatrix}
    -1 & 0 & 1\\
    -2 & 0 & 2\\
    -1 & 0 & 1\\
\end{bmatrix}$
  - $K_y = K_x^T$
- Identita $3\times3$
  - $Id=\begin{bmatrix}
    0 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 0\\\end{bmatrix}$
- Normalizovaný box blur $3\times3$
  - $K=\frac{1}{9}\begin{bmatrix}
    1 & 1 & 1\\
    1 & 1 & 1\\
    1 & 1 & 1\\\end{bmatrix}$
- Gaussian blur (aproximace) $3\times3$
  - $K=\frac{1}{16}\begin{bmatrix}
    1 & 2 & 1\\
    2 & 4 & 2\\
    1 & 2 & 1\\\end{bmatrix}$
- Doostření (sharpen) $3\times3$
  - $K=\begin{bmatrix}
    0 & -1 & 0\\
    -1 & 5 & -1\\
    0 & -1 & 0\\\end{bmatrix}$

## 10. Detekce objektů a rozpoznávání obličejů

- **Haar** (sliding window)
- **HOG** (sliding window)
- **LBP** (sliding window)
- **SIFT, SURF** (zájmové body)
- **CNNs** (hluboké učení)

### 10.1. Sliding window

1. Vstupní obraz je snímán *obdélníkovým oknem* v několika měřítcích.
2. Výsledkem procesu skenování je velké množství různých dílčích oken.
3. Z každého dílčího okna je extrahován *vektor příznaků*.
4. Tento vektor se použije jako vstup pro klasifikátor. Tento klasifikátor musí být předem natrénovaný.
5. Cílem je vytvořit *bounding box* okolo hledaného objektu.

### 10.2. Proces rozpoznávání

1. Detekce obličeje
2. Extrakce ROI
3. Extrakce příznaků
4. Fáze rozpoznávání (klasifikace)

### 10.3. Detekce obrazový zájmový bodů

- *facial landmark detection, keypoint detection*
- obrazové zájmové body mohou být použity k lepšímu výřezu obličeje a zlepšit tak rozpoznávání
- Facial landmarks can be used to align facial imagesto improve face recognition.
- můžeme odhadovat pózu hlavy - kam se člověk dívá
- nahrazení obličeje
- detekce zavřených očí

S jakými problémy musíme počítat?

- póza - obličej zepředu, z profilu, shora, zespodu atd.
- brýle, vlasy
- výrazy obličeje, mimika
- osvětlení, stíny, jas

### 10.4. Haar features (příznaky)

- Viola & Jones 2001
- hlavní myšlenka - obličeje mají podobné vlastnosti
  - oblast očí je tmavší něž líce, podobně čelo je obvykle světlejší než oči
  - ústa mají taky jiný jas
  - "nosní most" je světlejší než oči
- vezme se suma jasů v jednotlivých obdélnících a udělá se jejich rozdíl
- je nutná trénovací množina
- pro oči lze použít thresholding na bílou barvu

Jak funguje a co to je kaskádový klasifikátor?

- většina obrazu obvykle hledaný objekt neobsahuje, proto chceme tyto oblast co nejdříve zahodit
- kaskádový klasifikátor má několik fází, pokud je pravděpodobnost hledaného objektu v okně velká, tak se pokračuje do další fáze, jinak se okno zahodí (nebo něco takového)

#### 10.4.1. Kaskádová regrese (Cascaded Regression)

- tato metoda začíná z průměrného tvaru obličeje a iterativně posouvá zájmové body podle obrazových příznaků

![Cascaded Regression](figures/cascaded_regrassion.png)

#### 10.4.2. Knihovny pro detekci zájmový bodů obličeje

- **OpenCV**
- **DLIB**
- **MediaPipe** (Google)

## 11. Autonomí vozidla

Autonomní vozidlo *(AV - Autonomous Vehicle)* je vozidlo, které je schopné vnímat své okolí a bezpečně se pohybovat s malým nebo žádným lidským zásahem.

- pozemní vozidla
- drony (UAV)
- NOMARS - No Manning Required Ship

### 11.1. Senzory

- Lidar ("light detection and ranging"; např. firma Velodyne)
- Radar
- Kamery
- Mapy, GPS

Jak funguje LIDAR?

- Lidar používá laserové paprsky bezpečné pro oči, které "vidí" svět ve 3D (vytvářejí mračna 3D bodů) a poskytují počítačům prostorový obraz.
- Typický lidar emituje pulzy světla (vlnění) do svého okolí a toto vlnění se odráží od okolních objektů. Lidar tyto odražené vlny zaznamenává a počítá rozdíl v čase od vypuštění do zachycení.
- Tento proces se opakuje třeba milionkrát za sekundu.

Jaké má LIDAR výhody a nevýhody oproti kamerám?

- Nemůže detekovat barvy a interpretovat text, značky, semafory atd.
- Funguje velmi dobře i ve tmě.
- Přesnost.
- Vyšší cena a větší velikost (třeba nějaká "boule" na střeše)
- Poskytuje 3D obraz okolí.

Jaké výhody a nevýhody mají kamery?

- Rozpoznání barev, čtení dopravních značek.
- Moderní AI metody dokážou také vytvořit 3D obraz okolí (z dostatečného množství kamer).
- Vyžadují mnohem více výpočetního výkonu.
- Kamerové systémy jsou téměř neviditelné.
- Problémy za špatného světla.

### 11.2. Úrovně autonomních vozidel

- Level 0 - žádná automatizace
  - Veškeré řízení provádí řidič,ale vozidlo může pomáhat s detekcí mrtvého úhlu, varováním před čelní srážkou a varováním před opuštěním jízdního pruhu.

- Level 1 - asistenční systémy
  - Vozidlo může být vybaveno některými aktivními asistenčními funkcemi, ale řidič je stále zodpovědný za řízení. Mezi takové asistenční funkce, které jsou v dnešních vozidlech k dispozici, patří adaptivní tempomat, automatické nouzové brzdění a udržování v jízdním pruhu.

- Level 2 - částečná automatizace
  - Řidič musí být stále ve střehu a neustále sledovat okolí, ale funkce jízdních asistentů, které ovládají zrychlování, brzdění a řízení, mohou pracovat společně, takže řidič nemusí v určitých situacích nic zadávat. Mezi takové automatizované funkce, které jsou dnes k dispozici, patří například samočinné parkování a asistent pro jízdu v dopravní zácpě (jízda v režimu "stop and go traffic").

- Level 3 - podmíněná automatizace
  - Vozidlo může za určitých okolností samo vykonávat některé úkony řízení, ale řidič musí být vždy připraven převzít kontrolu nad řízením v rámci stanovené doby. Ve všech ostatních případech vykonává řízení člověk.

- Level 4 - vysoká automatizace
  - Jedná se o samořiditelné vozidlo. Stále však má sedadlo řidiče a běžné ovládací prvky. Přestože vozidlo může řídit a "vidět" samo, okolnosti, jako je geografická oblast, podmínky na silnici nebo místní zákony, mohou vyžadovat, aby osoba na místě řidiče převzala řízení.
  - Pár taxíků v Americe.

- Level 5 - úplná automatizace
  - Vozidlo je schopno plnit všechny jízdní funkce za všech podmínek prostředí a může být provozováno s lidmi na palubě.Lidé na palubě jsou cestující a nemusí se nikdy podílet na řízení. Volant je v tomto vozidle volitelný.

![Autonomous vehicles](figures/av.png)

## 12. OpenCV

Load image:

```cpp
cv::Mat src_8uc3_img = cv::imread( "images/lena.png", cv::IMREAD_COLOR );
cv::Mat src_8uc1_img = cv::imread( "images/lena.png", cv::IMREAD_GRAYSCALE );
```

- One pixel is represented by unsigned char (8 bits).

Conversion to gryscale:

```cpp
cv::cvtColor( src_8uc3_img, gray_8uc1_img, cv::COLOR_BGR2GRAY );
```

Conversion to float $([0,255] \rightarrow [0,1])$:

```cpp
gray_8uc1_img.convertTo( gray_32fc1_img, CV_32FC1, 1.0 / 255.0 );
```

Draw a rectangle:

```cpp
cv::rectangle(gray_8uc1_img,
              cv::Point(65, 84),
              cv::Point(75, 94),
              cv::Scalar(50),
              cv::FILLED);
```

<details><summary> Example: Access pixel values </summary>

- template method `cv::Mat.at<image_type>(int y, int x)`

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    int x = 0;
    int y = 0;

    // read grayscale value of a pixel, image represented using 8 bits
    uchar p1 = gray_8uc1_img.at<uchar>(y, x);

    // read grayscale value of a pixel, image represented using 32 bits
    float p2 = gray_32fc1_img.at<float>(y, x);

    // read color value of a pixel, image represented using 8 bits per color channel
    cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x);

    // print values of pixels
    printf("p1 = %d\n", p1);
    printf("p2 = %f\n", p2);
    printf("p3[0] = %d, p3[1] = %d, p3[2] = %d\n", p3[0], p3[1], p3[2]);

    // set pixel value to 0 (black)
    gray_8uc1_img.at<uchar>( y, x ) = 0;

    return 0;
}
```

</details>

<details><summary> Example: Creating gradient </summary>

```cpp
// Declare a variable to hold the gradient image with dimensions:
// width = 256 pixels, height = 50 pixels.
// Gray levels wil be represented using 8 bits (uchar).
cv::Mat gradient_8uc1_img( 50, 256, CV_8UC1 );

// For every pixel in image, 
// assign a brightness value
// according to the `x` coordinate.
// This wil create a horizontal gradient.
for ( int y = 0; y < gradient_8uc1_img.rows; y++ ) {
    for ( int x = 0; x < gradient_8uc1_img.cols; x++ ) {
        gradient_8uc1_img.at<uchar>( y, x ) = x;
    }
}

cv::imshow("Gradient", gradient_8uc1_img);
```

</details>
