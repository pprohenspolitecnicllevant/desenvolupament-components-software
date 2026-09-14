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
- Al 5149 els algorismes **es programen per dins** (NumPy + POO, amb la interfície
  `fit`/`predict`/`score` de scikit-learn) i després es contrasten amb la biblioteca.

## Unitats de treball

| UT | Títol | Estat |
|---|---|---|
| 1 | Introducció a xarxes neuronals supervisades: del perceptró al perceptró multicapa amb retropropagació | NB 1.1 (XOR, pingüins, llunes), NB 1.2 (MNIST) i diapositives |
| — | Xarxes neuronals no supervisades: k-means, SOM i xarxes de Hopfield | pendent |
| — | Llibreries de tercers, accés a dades, pla de proves i publicació de paquets | pendent |
| — | Anàlisi de dades amb NumPy, pandas i visualització amb Matplotlib i Seaborn | pendent |
| — | Integració i neteja de dades, procés KDD i algorismes d'aprenentatge automàtic amb llibreries | pendent |

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
