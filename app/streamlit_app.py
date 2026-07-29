"""Interactive Streamlit front-end for the BESS research pipeline."""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from bess_optimization.config import AnalysisConfig, BESSConfig, HedgeConfig, ScenarioConfig
from bess_optimization.data.entsoe_client import EntsoeSpotClient
from bess_optimization.data.schemas import validate_forward_quotes
from bess_optimization.pipelines.run_analysis import AnalysisResult, run_analysis
from bess_optimization.visualization.charts import (
    cycles_distribution_chart,
    dispatch_chart,
    forward_curve_chart,
    hedge_volumes_chart,
    pnl_boxplot,
    pnl_comparison_chart,
    risk_table_chart,
    scenario_paths_chart,
    scenario_quantiles_chart,
    soc_chart,
    spot_history_chart,
)

st.set_page_config(page_title="BESS Optimization", layout="wide")
st.title("BESS merchant revenue & futures hedge")
st.caption(
    "Outil de recherche — les données de démonstration sont synthétiques et le perfect "
    "foresight constitue une borne haute, pas une stratégie réalisable."
)

tabs = st.tabs(["Données", "Batterie", "Scénarios", "Dispatch", "Risque", "Hedging", "Export"])

with tabs[0]:
    source = st.radio("Source spot", ["Synthétique", "ENTSO-E"])
    market = st.text_input("Zone de marché", "FR")
    timezone = st.text_input("Timezone", "Europe/Paris")
    start = st.text_input("Début de l'analyse", "2027-01-01")
    end = st.text_input("Fin de l'analyse", "2027-01-14 23:00")
    history_start = st.text_input("Début historique", "2022-01-01")
    history_end = st.text_input("Fin historique", "2025-12-31 23:00")
    upload = st.file_uploader("CSV futures EEX fourni/licencié par l'utilisateur", type="csv")
    if source == "ENTSO-E" and not os.getenv("ENTSOE_API_KEY"):
        st.warning("ENTSOE_API_KEY absent : choisissez le mode synthétique ou configurez la clé.")

with tabs[1]:
    c1, c2, c3 = st.columns(3)
    charge_mw = c1.number_input("Charge maximale (MW)", 0.1, value=50.0)
    discharge_mw = c1.number_input("Décharge maximale (MW)", 0.1, value=50.0)
    capacity = c1.number_input("Capacité (MWh)", 0.1, value=100.0)
    eta_c = c2.number_input("Rendement de charge", 0.01, 1.0, value=0.9434)
    eta_d = c2.number_input("Rendement de décharge", 0.01, 1.0, value=0.9434)
    degradation = c2.number_input("Dégradation (€/MWh déchargé)", 0.0, value=3.0)
    initial_soc = c3.number_input("SOC initial (MWh)", 0.0, value=50.0)
    final_soc = c3.number_input("SOC final (MWh)", 0.0, value=50.0)
    cycles_day = c3.number_input("Cycles équivalents max/jour", 0.1, value=1.5)
    st.caption(f"Rendement aller-retour : {eta_c * eta_d:.2%}")

with tabs[2]:
    method = st.selectbox("Modèle", ["bootstrap", "factor"])
    n_scenarios = st.slider("Nombre de scénarios", 5, 100, 20)
    seed = st.number_input("Seed", value=42, step=1)
    block_days = st.slider("Taille des blocs bootstrap (jours)", 1, 21, 7)

with tabs[3]:
    dispatch_mode = st.selectbox(
        "Mode d'optimisation",
        ["expected", "perfect_foresight"],
        format_func=lambda value: (
            "Dispatch commun sur courbe attendue"
            if value == "expected"
            else "Perfect foresight (borne haute)"
        ),
    )
    quantile_enabled = st.checkbox("Optimiser le dispatch commun sur un quantile")
    quantile = st.slider("Quantile", 0.0, 1.0, 0.5) if quantile_enabled else None
    run = st.button("Lancer l'analyse", type="primary")

with tabs[4]:
    minimum_target = st.number_input("Revenu minimum cible (€)", value=0.0)

with tabs[5]:
    hedge_enabled = st.checkbox("Optimiser le hedge", value=True)
    hedge_products = st.selectbox("Produits", ["both", "baseload", "peakload"])
    hedge_objective = st.selectbox("Critère", ["min_variance", "mean_variance"])
    risk_lambda = st.number_input("Lambda de risque", min_value=0.0, value=0.0)
    volume_bound = st.number_input("Limite absolue par produit (MW)", 0.0, value=50.0)
    volume_step = st.number_input("Pas de cotation (MW, 0 = continu)", 0.0, value=0.0)

if run:
    try:
        battery = BESSConfig(
            max_charge_mw=charge_mw,
            max_discharge_mw=discharge_mw,
            energy_capacity_mwh=capacity,
            max_soc_mwh=capacity,
            initial_soc_mwh=initial_soc,
            final_soc_mwh=final_soc,
            charge_efficiency=eta_c,
            discharge_efficiency=eta_d,
            degradation_cost_eur_mwh=degradation,
            max_equivalent_cycles_per_day=cycles_day,
        )
        config = AnalysisConfig(
            market=market,
            timezone=timezone,
            start=start,
            end=end,
            historical_start=history_start,
            historical_end=history_end,
            dispatch_mode=dispatch_mode,
            dispatch_quantile=quantile,
            minimum_revenue_target_eur=minimum_target,
            battery=battery,
            scenarios=ScenarioConfig(
                method=method, n_scenarios=n_scenarios, seed=int(seed), block_days=block_days
            ),
            hedge=HedgeConfig(
                enabled=hedge_enabled,
                products=hedge_products,
                objective=hedge_objective,
                risk_aversion_lambda=risk_lambda,
                min_volume_mw=-volume_bound,
                max_volume_mw=volume_bound,
                volume_step_mw=volume_step or None,
            ),
        )
        history = None
        if source == "ENTSO-E":
            history = EntsoeSpotClient().fetch_day_ahead_prices(
                market, history_start, history_end, timezone
            )
        quotes = validate_forward_quotes(pd.read_csv(upload), timezone) if upload else None
        with st.spinner("Génération, MILP et hedge en cours…"):
            st.session_state["analysis"] = run_analysis(
                config, history=history, forward_quotes=quotes, export=False
            )
    except Exception as exc:
        st.exception(exc)

result: AnalysisResult | None = st.session_state.get("analysis")
if result is not None:
    with tabs[0]:
        st.plotly_chart(forward_curve_chart(result.forward_quotes), use_container_width=True)
        st.plotly_chart(
            spot_history_chart(result.history.iloc[-24 * 30 :]), use_container_width=True
        )
    with tabs[2]:
        st.plotly_chart(scenario_paths_chart(result.scenarios), use_container_width=True)
        st.plotly_chart(scenario_quantiles_chart(result.scenarios), use_container_width=True)
    with tabs[3]:
        st.plotly_chart(
            dispatch_chart(result.representative_dispatch.schedule), use_container_width=True
        )
        st.plotly_chart(
            soc_chart(result.representative_dispatch.schedule), use_container_width=True
        )
    with tabs[4]:
        st.plotly_chart(cycles_distribution_chart(result.pnl_by_scenario), use_container_width=True)
        st.plotly_chart(risk_table_chart(result.risk), use_container_width=True)
    with tabs[5]:
        if result.hedge:
            st.plotly_chart(
                pnl_comparison_chart(result.hedge.unhedged_pnl_eur, result.hedge.hedged_pnl_eur),
                use_container_width=True,
            )
            st.plotly_chart(
                pnl_boxplot(result.hedge.unhedged_pnl_eur, result.hedge.hedged_pnl_eur),
                use_container_width=True,
            )
            st.plotly_chart(hedge_volumes_chart(result.hedge.volumes_mw), use_container_width=True)
            st.dataframe(result.hedge.product_correlations)
    with tabs[6]:
        st.download_button(
            "Télécharger le P&L (CSV)",
            result.pnl_by_scenario.to_csv().encode(),
            "pnl_by_scenario.csv",
            "text/csv",
        )
        parquet_buffer = io.BytesIO()
        result.scenarios.prices.to_parquet(parquet_buffer)
        st.download_button(
            "Télécharger les scénarios (Parquet)",
            parquet_buffer.getvalue(),
            "price_scenarios.parquet",
            "application/octet-stream",
        )
else:
    with tabs[6]:
        st.info("Lancez d'abord une analyse dans l'onglet Dispatch.")
