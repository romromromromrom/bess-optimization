# BESS Optimization

Outil Python de recherche quantitative pour estimer la valeur marchande future d'une
batterie, optimiser son dispatch spot et réduire le risque du P&L avec des futures
Calendar Baseload et Peakload.

Le projet fonctionne immédiatement avec des données **entièrement synthétiques**. Il sait
aussi lire des prix day-ahead ENTSO-E et des exports CSV de courbes futures obtenus
légalement par l'utilisateur. Il ne fournit ni accès direct ni contournement aux données
EEX.

## Démarrage rapide

Python 3.12 et un compilateur ne sont normalement pas nécessaires : `highspy` distribue
des roues binaires sur les plateformes courantes.

```bash
cd bess-optimization
# Debian/Ubuntu uniquement, si `venv`/`ensurepip` est absent :
# sudo apt install python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

bess-opt run-all --config configs/example.yaml
streamlit run app/streamlit_app.py
pytest
```

Dans le checkout ayant servi à la validation, un environnement existe déjà dans le
répertoire parent ; on peut donc aussi lancer `source ../.venv/bin/activate`.

La démonstration utilise 14 jours et 20 scénarios afin de résoudre rapidement les MILP.
Les résultats sont écrits dans `data/processed/demo/`.

## Objectif économique

La chaîne :

1. estime une forme horaire à partir de l'historique spot ;
2. transforme les blocs Calendar forward en courbe horaire centrale ;
3. génère des trajectoires de prix conservant saisonnalité, dépendance temporelle, prix
   négatifs et pointes ;
4. optimise les achats/ventes physiques du BESS ;
5. valorise les décisions dans chaque scénario et calcule les risques ;
6. construit des payoffs futures et choisit les MW signés réduisant la variance du P&L.

Deux modes de dispatch sont fondamentalement différents :

- `expected` optimise une seule décision sur la moyenne (ou un quantile) des scénarios,
  puis valorise exactement cette décision sur toutes les trajectoires. C'est le mode
  approprié pour étudier une vraie distribution ex ante sans look-ahead.
- `perfect_foresight` réoptimise chaque scénario en connaissant tous ses prix futurs.
  **C'est une borne haute de la valeur d'arbitrage, pas une stratégie réalisable.**

L'architecture réserve une interface à un futur modèle stochastique à deux étapes et à
une résolution parallèle/par fenêtres.

## Architecture

```text
src/bess_optimization/
├── config.py                 # paramètres Pydantic et YAML
├── data/                     # ENTSO-E, CSV forward, schémas, synthétique
├── curves/                   # shaping horaire et courbe centrale
├── scenarios/                # bootstrap, PCA/facteurs, diagnostics
├── battery/                  # MILP Pyomo, résolution HiGHS, métriques
├── valuation/                # P&L par scénario, VaR/CVaR
├── hedging/                  # produits, payoffs, hedge quadratique borné
├── visualization/            # graphiques Plotly
├── pipelines/                # orchestration bout en bout
└── cli.py                    # commandes `bess-opt`
```

`app/streamlit_app.py` contient l'interface à sept onglets. `tests/` contient des tests
unitaires physiques, financiers et statistiques.

## Hypothèses de courbe et scénarios

Un future Calendar ne détermine pas chacun des 8 760 prix horaires. La courbe horaire est
donc une **construction sous hypothèses** :

- Peakload = lundi-vendredi, 08:00 inclus à 20:00 exclu, heure locale ;
- profils mois / jour ouvré-week-end / heure estimés sur l'historique ;
- niveau off-peak implicite calculé pour respecter simultanément les moyennes Baseload et
  Peakload ;
- corrections séparées par année et profil pour rebaser les scénarios.

Le bootstrap échantillonne des blocs contigus de résidus historiques. Le modèle factoriel
applique une PCA aux variations de snapshots Calendar, puis ajoute un bruit horaire AR(1).
Une classe d'extension documente le point d'entrée HMM/régimes. Les scénarios sont
reproductibles par `seed` et enregistrables en Parquet.

Sur un horizon de démonstration partiel, les payoffs de futures sont calculés sur les
heures de livraison qui chevauchent cet horizon. Une analyse Calendar de production doit
simuler toute l'année de livraison.

## Convention batterie et unités

Les puissances réseau sont en MW, le SOC en MWh, les prix en €/MWh et `Δt` en heures :

```text
SOC[t+1] = SOC[t] + η_charge × P_charge[t] × Δt
                  - P_decharge[t] × Δt / η_decharge
```

Le rendement aller-retour vaut `η_charge × η_decharge`. L'exemple utilise
`0.9434 × 0.9434 ≈ 89 %`. Les énergies facturées au marché sont les puissances côté réseau
multipliées par `Δt`. Un cycle équivalent est ici l'énergie réseau déchargée divisée par
la capacité nominale.

Le MILP impose limites de puissance et d'énergie, modes binaires exclusifs, états initial
et final, disponibilité, rendements, coût variable, dégradation et plafond de cycles.
Une durée minimale de mode est facultative.

## Futures et convention de signe

Pour un produit et un scénario, le payoff unitaire est :

```text
payoff_long_1_MW = heures_de_livraison × (prix_règlement - prix_entrée)
P&L_total = P&L_spot_BESS + somme(volume_signé_MW × payoff_long_1_MW)
```

Un volume positif est **long**, un volume négatif est **short**. Le prix de règlement est
la moyenne des heures du profil (toutes les heures pour Baseload, heures Peak pour
Peakload). L'optimiseur minimise la variance avec bornes, pas de volume facultatif et
faible ridge de stabilisation, ou optimise `espérance - lambda × variance`.

VaR 95 % et CVaR 95 % sont exprimées comme pertes positives :
`VaR95 = -P5(P&L)` et `CVaR95 = -moyenne(P&L | P&L ≤ P5)`.

## Données

### ENTSO-E

Ne commitez jamais la clé :

```bash
export ENTSOE_API_KEY="votre-token"
bess-opt fetch-entsoe --start 2024-01-01 --end 2025-01-01 \
  --zone FR --output data/raw/entsoe_spot.parquet
```

Le client accepte une zone, des dates, une timezone et une sortie horaire ou
quart-horaire au niveau Python. Les jours de changement d'heure restent timezone-aware.

### Forward EEX fourni par l'utilisateur

Le chargeur CSV accepte plusieurs snapshots et exige :

```text
observation_date, delivery_start, delivery_end, product, load_type,
price_eur_mwh, volume_mw, currency, market
```

`load_type` vaut `baseload` ou `peakload`. Un exemple **illustratif, non issu d'EEX** se
trouve dans `data/sample/eex_forward_example.csv`. Les données EEX peuvent être soumises
à licence ; l'utilisateur reste responsable de ses droits d'utilisation.

## CLI

```bash
bess-opt generate-data --config configs/example.yaml
bess-opt fetch-entsoe --start 2024-01-01 --end 2025-01-01 --zone FR
bess-opt generate-scenarios --config configs/example.yaml
bess-opt optimize-dispatch --config configs/example.yaml
bess-opt optimize-hedge --config configs/example.yaml
bess-opt run-all --config configs/example.yaml
```

`optimize-dispatch` et `optimize-hedge` consomment le fichier de scénarios créé par
`generate-scenarios`. `run-all` orchestre et exporte tout.

## Qualité et tests

```bash
pytest
ruff check src tests app
ruff format --check src tests app
mypy src
```

Les tests couvrent conservation du SOC, rendements, limites, exclusivité, indisponibilité,
prix plat, arbitrage bas/haut, unités, reproductibilité, dimensions, absence de look-ahead
du dispatch commun et réduction de variance du hedge. Si HiGHS est absent, la résolution
échoue avec un message indiquant d'installer `highspy`.

## Limites et avertissement

- Les résultats ne constituent **pas une recommandation de trading ou
  d'investissement**.
- Le perfect foresight surestime structurellement les revenus accessibles.
- La courbe horaire dépend d'hypothèses statistiques et calendaires.
- Le modèle ne représente pas encore intraday, réserves, imbalance, rampes, auxiliaires,
  fiscalité, collateral/margining, coûts fixes ou vieillissement électrochimique détaillé.
- Le modèle factoriel MVP est parcimonieux ; peu de snapshots impliquent une calibration
  fragile.
- Plusieurs années × centaines de scénarios × MILP horaires doivent être traités par
  fenêtres ou en parallèle. Le pipeline de démonstration limite volontairement la charge.
- Le settlement Calendar est approché par la moyenne spot simulée ; les règles exactes du
  contrat et du fournisseur de données doivent être vérifiées avant usage réel.

## Améliorations prioritaires

1. optimisation stochastique à deux étapes et politique rolling horizon ;
2. résolution parallèle et décomposition par fenêtres avec continuité du SOC ;
3. HMM/régimes extrêmes, dépendance météo/fondamentaux et validation hors échantillon ;
4. courbes mensuelles/trimestrielles et contraintes de liquidité/bid-ask ;
5. vieillissement par profondeur de cycle et température ;
6. backtesting complet, suivi des données et artefacts de calibration.
