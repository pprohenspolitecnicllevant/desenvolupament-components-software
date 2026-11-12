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
- **Res de codi que simuli neurones a mà** (la neurona de McCulloch i Pitts, la suma
  ponderada i la funció step amb NumPy, xarxes amb pesos triats a mà…). Aquests
  conceptes s'expliquen a les diapositives i, al notebook, amb text i taules. El
  codi del notebook fa servir les biblioteques.
- **Els notebooks van de pressa fins al problema que motiva la unitat** (a la UT5,
  la XOR) i s'hi estenen a partir d'aquí. La història i els precedents, a les
  diapositives.
- **Pauses «Ara tu» intercalades** (CONVENCIONS §4): exercicis curts just després
  de la secció que practiquen, perquè l'explicació no s'allargui sense que
  l'alumnat hi posi les mans. Al final, només els exercicis que demanen el
  notebook sencer.
- **Res de conceptes implícits.** Quan el codi fa servir un concepte que la unitat
  no explica (per exemple, `stratify` a `train_test_split`), el notebook hi fa un
  **incís** la primera vegada que surt: què és, per què cal i un exemple amb números
  reals. A les diapositives, tots aquests conceptes van al **glossari** del final.
- **Conjunts de dades públics** a les UT5–UT9 (pingüins, `make_moons`, California
  Housing, MNIST, Fashion-MNIST, CIFAR-10, Mall Customers, SMS Spam Collection…),
  sense cas conductor. Cal revisar-ne la llicència.
- **Només Python** a les UT5–UT8. La resta del mòdul treballa Python i Node.js costat
  a costat, però aquí les biblioteques de referència no tenen un equivalent madur a
  Node.js.
- Cal que funcioni **en CPU i a Colab**. L'alumnat té GPU: la T4 de Colab, CUDA als ordinadors de l'aula
  (RTX 3050) i, de manera puntual, una DGX Spark. Les missions que la fan servir funcionen també sense,
  amb una versió reduïda o amb un model ja entrenat.

## Unitats de treball

La numeració és la de la programació didàctica: un full de Google compartit, «5149.
Desenvolupament de components software…». Al docent li corresponen les **UT5–UT9**.
Les UT1–UT4 (modelització d'algorismes i POO) són d'un altre docent i no es toquen.
La UT10 és la FEMPO.

| UT | Títol | h | Eines | Criteris d'avaluació | Estat |
|---|---|---|---|---|---|
| 5 | Introducció a xarxes neuronals supervisades: del perceptró al perceptró multicapa amb retropropagació | 28 | scikit-learn | 3.a, 3.f | carpeta `UT01-...`: NB 1.1 (XOR, pingüins, llunes; 5 pauses «Ara tu» + 2 finals), NB 1.2 (MNIST; 4 pauses «Ara tu» + 4 finals), diapositives (47, amb glossari de 32 conceptes al final) i tres animacions del bucle d'entrenament (`animacio_xor_backprop.html`; `animacio_pinguins_backprop.html`: Chinstrap o no; `animacio_pinguins_especies.html`: les tres espècies amb softmax, tres mesures fixes d'entrada i un gràfic 3D giratori amb les regions de decisió i els plans de les neurones ocultes, que són configurables), adaptats a scikit-learn. Exercicis de consolidació a `consolidacio_xarxa_electrica/` (Electrical Grid Stability i Fashion-MNIST; 8 exercicis i 5 extra, unes 4 h 30); el generador dels tres notebooks és al repositori privat |
| 6 | Xarxes neuronals convolucionals i introducció a la visió per computador | 24 | PyTorch, torchvision, Ultralytics | 3.a | carpeta `UT06-...`: NB 6.1–6.6 (La lupa, El laboratori, Crash test, A espatlles de gegants, Més enllà de classificar i el projecte; 28 missions «Ara tu», una missió extra amb CLIP al NB 6.5 i 9 finals), diapositives (32: 29 + glossari de 25 conceptes en 3) i animació `animacio_lupa_cnn.html`. Solucionaris al repositori privat. Pendent: revisió del docent i prova a Colab |
| 7 | Xarxes neuronals no supervisades: k-means, SOM i xarxes de Hopfield | 24 | scikit-learn, MiniSom, classe Hopfield | 3.b–3.e | pendent |
| 8 | Llibreries de tercers, accés a dades, anàlisi amb NumPy, pandas, Matplotlib i Seaborn, KDD i **inferència amb el model entrenat** | 24 | pandas, requests, joblib, pytest | 4.a–4.f, 2.h–2.k | pendent |
| 9 | Documentació professional del codi, dels components i de les proves | 20 | docstrings, pdoc/Sphinx | 5.a–5.e | pendent |

La carpeta `UT01-...` del repositori correspon a la **UT5** de la programació. No es
reanomena, perquè les URL raw de MNIST hi apunten.

## La UT6 (`UT06-Xarxes_convolucionals/`)

Vuit sessions per missions: blocs de 10 minuts d'explicació com a màxim i la resta de
treball guiat. L'alumnat **no programa les xarxes**: les entén, les fa servir i les
mesura. El BOE parla de programar-les, però es manté el criteri d'aquest mòdul.

Tres excepcions a les convencions, només per a aquesta unitat:

| Convenció | Canvi a la UT6 | Motiu |
|---|---|---|
| Unes 3 pauses «Ara tu» per notebook | 5–6 missions per notebook | L'aprenentatge recau en els exercicis |
| Càrrega alta de markdown | Cada cel·la de codi manté l'explicació, però curta | Explicacions breus |
| Res de `files.upload()` | Permès només per a fotos pròpies: NB 6.4 i 6.5 (opcional) i projecte | El projecte necessita dades pròpies |

- **Peça comuna.** `make_cnn(in_channels, n_classes, image_size=28, filters=(16, 32),
  pooling=True, global_pool=False, batch_norm=False, dropout=0.0)` crea la xarxa i
  `train_model` l'entrena. Totes dues, amb `make_loaders`, `accuracy` i `plot_history`,
  van a la cel·la «No cal modificar-la» de cada notebook, sense dependre de cap fitxer.
- **Dades.** `prepare_data.py` genera les còpies de `fashion_mnist/`, `quickdraw/`,
  `pedra_paper_tisora/` i `eurosat/`, i `fotos/` són imatges amb llicència oberta. Cada
  carpeta porta `ATRIBUCIO.md`. CIFAR-10 es baixa de Hugging Face i MNIST és el de la UT5.
- **Models.** `train_models.py` entrena `models/cifar_cnn.pt` i `models/cifar_cnn_aug.pt`,
  que el NB 6.3 carrega amb `weights_only=True`. `--export-json` exporta els pesos de la CNN
  de l'animació, que porta els pesos incrustats i no s'hi entrena res, a diferència de les
  altres `animacio_*`. És l'únic lloc de la unitat, amb les diapositives, amb filtres triats a mà.
- **YOLO11n** (Ultralytics, AGPL-3.0) al NB 6.5: ja entrenat a les missions 1–3, i reentrenat amb *African
  Wildlife* (d'Ultralytics) a la missió 4 si hi ha GPU. Sense GPU es carrega `models/fauna_yolo11n.pt`,
  entrenat 10 epochs (mAP50 0,93), perquè a la CPU no aprèn res que valgui en uns minuts.
- **Transferència d'estil** (VGG19, Gatys 2015) a la missió 5 del NB 6.5, amb els quadres de domini públic
  d'`estils/`. Sense GPU, imatges de 224 píxels i 120 passos.
- **CLIP** (`openai/clip-vit-base-patch32`, amb `transformers`) a la missió extra del NB 6.5: classifica
  zero-shot amb les etiquetes que escriu l'alumnat. Només com a model ja entrenat i sense preguntes sobre persones.
- **Les diapositives són públiques:** només hi van números del text dels notebooks o de
  cel·les donades. Els resultats de les missions no. Ho comprova `tests/test_slides.py`
  del repositori privat.
- **Fotos:** només objectes. Sense cares, persones, matrícules ni dades privades, tret
  de les fotos que ja es donen. Les dades del projecte no es pugen a cap repositori.
- **GPU.** La GPU s'explica al NB 6.1 §2 amb «La cursa» (`race`, codi donat, no és cap missió): 1 epoch de
  l'MLP del NB 1.2 amb scikit-learn i amb PyTorch i de la CNN, a la CPU i a la GPU si n'hi ha. El missatge: la GPU
  surt a compte amb xarxes que fan molts càlculs; amb un MLP petit, scikit-learn guanya o empata. El fine-tuning és la missió 6 del NB 6.4 (sense GPU, imatges de 112 píxels i 3 epochs) i un pas
  opcional del projecte (NB 6.6, pas 4 bis). La competició del NB 6.2 té una lliga CPU (10.000 imatges,
  150.000 pesos, 5 epochs) i una lliga GPU (40.000 imatges, 1.000.000 de pesos, 10 epochs).
- **Temps:** cap entrenament de més de 2 minuts a la CPU de Colab; els notebooks avisen
  de la durada amb `⏱`. Els temps se suposen 3 vegades els d'un M1 fins que es faci la prova.
  Amb GPU s'admeten uns minuts (YOLO, lliga GPU). Els temps de GPU s'han mesurat a la GPU d'un M1, no a la T4.

## Regles ràpides

- **Text en català** (markdown dels notebooks i diapositives).
- **Termes tècnics d'ML en anglès**, sense traduir: decision tree, random forest,
  k-nearest neighbors, epoch, batch / mini-batch, learning rate, overfitting, step,
  loss, gradient descent, backpropagation («una epoch», «el learning rate», «la loss»).
  Queden en català capa oculta, pesos, biaix, característica, funció d'activació,
  regressió, perceptró i llavor.
- **Codi dels notebooks**: identificadors en anglès; comentaris, docstrings,
  missatges (`print`), etiquetes dels gràfics i noms de columnes de les taules en
  **català**, amb els termes d'ML en anglès com a la prosa (loss, epoch, step,
  learning rate, batch…). Els exercicis diuen `# El teu codi aquí`.
- Les **figures de les diapositives** van amb etiquetes en català.
- Dades per **URL raw de GitHub** (els pingüins es carreguen del repositori del 5134).
  MNIST és una còpia de `fetch_openml("mnist_784")` a `UT01-.../mnist/mnist.npz`
  (uint8, 60.000 + 10.000), per no dependre d'OpenML a Colab.
- Cada cel·la de codi porta la seva cel·la d'explicació. Referències creuades
  explícites al 5134 quan hi ha relació.
- Els números que cita el text han de coincidir amb l'execució real: **executa el
  notebook sencer abans de donar-lo per bo**.
- **Aquest repositori és públic i l'alumnat el llegeix.** Els exercicis hi van amb
  l'enunciat, però **cap solució**: els solucionaris s'escriuen al repositori
  privat bessó (`../desenvolupament-components-software-solucions`) i arriben a
  l'alumnat per Google Classroom després del termini d'entrega. El `.gitignore`
  bloqueja `*_solucions*.ipynb` com a xarxa de seguretat.

## Diapositives

HTML autocontingut amb el mateix sistema visual que les del 5134 (16:9,
Georgia + Helvetica Neue, lacre `#0F1B33` / `#1C7293` / `#E8871E`) i gràfics SVG
inserits. El PDF s'obté imprimint l'HTML amb Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer --print-to-pdf=UTn.pdf UTn.html
```

Les diapositives acaben amb el debat de classe i, darrere, un **glossari** dels
conceptes que surten de passada a la unitat (diapositives i notebooks): blocs de la
classe `.gloss` en tres columnes, ordenats alfabèticament, amb una definició curta de
què és i per què importa, on surt (diapositiva, secció o exercici del notebook) i on
es treballa a fons al 5134. Els conceptes centrals de la unitat no hi van.

## Estructura

```
UTnn-Nom_de_la_unitat/
  NB_u_n_titol.ipynb
  UTn_titol.html      font de les diapositives
  UTn_titol.pdf
  animacio_*.html     animacions per projectar a classe: HTML autocontingut, el model
                      s'entrena en JavaScript dins la pàgina; s'enllacen des de la diapositiva
  <dataset>/          dades que carreguen els notebooks per URL raw
  figures/            figures estàtiques (PNG) que els notebooks mostren per URL raw,
                      en lloc de codi que les dibuixi; make_figures.py les regenera
```
