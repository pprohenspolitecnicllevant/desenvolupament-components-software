# Context del projecte

Aquest repositori és **material docent**, no programari. Conté els notebooks i les
diapositives del mòdul professional **5149 — Desenvolupament de components software
per a sistemes d'aprenentatge automàtic** (cicle IAD41, especialització en IA i Big
Data, Politècnic Llevant, curs 2026-27). Normativa: BOE-A-2026-5869 (90 h).

El mòdul germà és el **5134** (`../disseny-avaluacio-models-ml`), que imparteix el
mateix docent en paral·lel. Abans de tocar res, llegeix-ne `docs/CONVENCIONS.md`:
aquí se segueixen les mateixes convencions amb les excepcions de sota.

## Repartiment entre 5134 i 5149

- **Totes les xarxes neuronals són del 5149.** S'han tret del 5134 (on hi havia la
  UT9 «Primer contacte amb xarxes neuronals»). El 5134 es queda amb els algorismes
  clàssics; el 5149 tracta les xarxes com una **família diferent** i ho fa explícit.
- El fil conductor: *als algorismes clàssics les característiques les prepara la
  persona; a les xarxes, les capes ocultes les aprenen*.
## Enfocament: xarxes d'alt nivell, amb biblioteques

L'alumnat no té la base matemàtica per programar les xarxes per dins. Per això:

- Les xarxes **s'entrenen amb biblioteques**, com a components d'alt nivell amb la
  interfície `fit`/`predict`/`score`. **No es programen des de zero** amb NumPy.
  - **scikit-learn**: `Perceptron`, `MLPClassifier`, `MLPRegressor`, `KMeans`,
    `Pipeline`, `GridSearchCV`.
  - **PyTorch** (només per a les CNN): el bucle d'entrenament es dona com a
    **plantilla comentada**. L'alumnat en canvia l'arquitectura, els
    hiperparàmetres i les dades, però no l'escriu.
  - **MiniSom** per als SOM.
  - Per a **Hopfield**, una classe proporcionada pel professorat amb `fit`/`predict`.
    Cap biblioteca mantinguda la inclou.
- Els conceptes s'expliquen de manera **visual i intuïtiva**, sense derivades ni
  desenvolupament matemàtic. Alguns exemples:
  - la pèrdua com a distància a l'encert;
  - el descens del gradient com a baixar una muntanya a les palpentes;
  - la retropropagació com a repartiment de l'error cap enrere;
  - la convolució com una lupa que recorre la imatge.
- On aporti, es fan servir fronteres de decisió, corbes de pèrdua i animacions.
- **Conjunts de dades públics** a les UT5–UT9 (pingüins, `make_moons`, California
  Housing, MNIST, Fashion-MNIST, CIFAR-10, Mall Customers, SMS Spam Collection…),
  sense cas conductor. Cal revisar-ne la llicència.
- **Només Python** a les UT5–UT8. La resta del mòdul treballa Python i Node.js costat
  a costat, però aquí les biblioteques de referència no tenen un equivalent madur a
  Node.js.
- Cal que funcioni **en CPU i a Colab**. La GPU de Colab és opcional.

## Unitats de treball

La numeració és la de la programació didàctica: un full de Google compartit, «5149.
Desenvolupament de components software…». Al docent li corresponen les **UT5–UT9**.
Les UT1–UT4 (modelització d'algorismes i POO) són d'un altre docent i no es toquen.
La UT10 és la FEMPO.

| UT | Títol | h | Eines | Criteris d'avaluació | Estat |
|---|---|---|---|---|---|
| 5 | Introducció a xarxes neuronals supervisades: del perceptró al perceptró multicapa amb retropropagació | 28 | scikit-learn | 3.a, 3.f | carpeta `UT01-...`: NB 1.1 (XOR, pingüins, llunes), NB 1.2 (MNIST) i diapositives, adaptats a scikit-learn |
| 6 | Xarxes neuronals convolucionals i introducció a la visió per computador | 24 | PyTorch, torchvision | 3.a | pendent |
| 7 | Xarxes neuronals no supervisades: k-means, SOM i xarxes de Hopfield | 24 | scikit-learn, MiniSom, classe Hopfield | 3.b–3.e | pendent |
| 8 | Llibreries de tercers, accés a dades, anàlisi amb NumPy, pandas, Matplotlib i Seaborn, KDD i **inferència amb el model entrenat** | 24 | pandas, requests, joblib, pytest | 4.a–4.f, 2.h–2.k | pendent |
| 9 | Documentació professional del codi, dels components i de les proves | 20 | docstrings, pdoc/Sphinx | 5.a–5.e | pendent |

La carpeta `UT01-...` del repositori correspon a la **UT5** de la programació. No es
reanomena, perquè les URL raw de MNIST hi apunten.

## Regles ràpides

- **Text en català** (markdown dels notebooks i diapositives).
- **Codi en anglès**: identificadors, comentaris, docstrings, missatges i etiquetes
  dels gràfics dins del notebook.
- Les **figures de les diapositives** van amb etiquetes en català.
- Dades per **URL raw de GitHub** (els pingüins es carreguen del repositori del 5134).
  MNIST és una còpia de `fetch_openml("mnist_784")` a `UT01-.../mnist/mnist.npz`
  (uint8, 60.000 + 10.000), per no dependre d'OpenML a Colab.
- Cada cel·la de codi porta la seva cel·la d'explicació. Referències creuades
  explícites al 5134 quan hi ha relació.
- Els números que cita el text han de coincidir amb l'execució real: **executa el
  notebook sencer abans de donar-lo per bo**.

## Diapositives

HTML autocontingut amb el mateix sistema visual que les del 5134 (16:9,
Georgia + Helvetica Neue, lacre `#0F1B33` / `#1C7293` / `#E8871E`) i gràfics SVG
inserits. El PDF s'obté imprimint l'HTML amb Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer --print-to-pdf=UTn.pdf UTn.html
```

## Estructura

```
UTnn-Nom_de_la_unitat/
  NB_u_n_titol.ipynb
  UTn_titol.html      font de les diapositives
  UTn_titol.pdf
  <dataset>/          dades que carreguen els notebooks per URL raw
```
