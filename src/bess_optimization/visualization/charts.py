"""Plotly chart factory functions with explicit business units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bess_optimization.scenarios.base import PriceScenarioSet


def forward_curve_chart(quotes: pd.DataFrame) -> go.Figure:
    latest = (
        quotes.sort_values("observation_date")
        .groupby(["delivery_start", "load_type"], as_index=False)
        .tail(1)
    )
    fig = px.line(
        latest,
        x="delivery_start",
        y="price_eur_mwh",
        color="load_type",
        markers=True,
        title="Courbe forward Calendar Baseload / Peakload",
        labels={
            "delivery_start": "Début de livraison",
            "price_eur_mwh": "Prix (€/MWh)",
            "load_type": "Profil",
        },
    )
    return fig


def spot_history_chart(history: pd.Series) -> go.Figure:
    return px.line(
        history.rename("Prix spot").reset_index(),
        x="index",
        y="Prix spot",
        title="Historique spot (données synthétiques ou ENTSO-E)",
        labels={"index": "Date", "Prix spot": "Prix (€/MWh)"},
    )


def scenario_paths_chart(scenarios: PriceScenarioSet, max_paths: int = 20) -> go.Figure:
    frame = scenarios.prices.iloc[:, :max_paths]
    fig = go.Figure()
    for column in frame:
        fig.add_scatter(x=frame.index, y=frame[column], mode="lines", name=column, opacity=0.35)
    fig.update_layout(
        title="Trajectoires horaires de prix",
        xaxis_title="Date",
        yaxis_title="Prix (€/MWh)",
    )
    return fig


def scenario_quantiles_chart(scenarios: PriceScenarioSet) -> go.Figure:
    prices = scenarios.prices
    q05, q50, q95 = (prices.quantile(q, axis=1) for q in (0.05, 0.5, 0.95))
    mean = prices.mean(axis=1)
    fig = go.Figure()
    fig.add_scatter(x=prices.index, y=q95, line={"width": 0}, name="P95")
    fig.add_scatter(
        x=prices.index,
        y=q05,
        fill="tonexty",
        line={"width": 0},
        name="P5–P95",
    )
    fig.add_scatter(x=prices.index, y=q50, name="Médiane")
    fig.add_scatter(x=prices.index, y=mean, name="Moyenne")
    fig.update_layout(
        title="Moyenne et quantiles des scénarios",
        xaxis_title="Date",
        yaxis_title="Prix (€/MWh)",
    )
    return fig


def dispatch_chart(schedule: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=schedule.index, y=-schedule["charge_mw"], name="Charge")
    fig.add_bar(x=schedule.index, y=schedule["discharge_mw"], name="Décharge")
    fig.update_layout(
        barmode="relative",
        title="Programme de charge / décharge",
        xaxis_title="Date",
        yaxis_title="Puissance réseau (MW)",
    )
    return fig


def soc_chart(schedule: pd.DataFrame) -> go.Figure:
    return px.line(
        schedule.reset_index(),
        x="index",
        y="soc_end_mwh",
        title="État de charge",
        labels={"index": "Date", "soc_end_mwh": "SOC (MWh)"},
    )


def cycles_distribution_chart(pnl: pd.DataFrame) -> go.Figure:
    return px.histogram(
        pnl,
        x="equivalent_cycles",
        title="Distribution des cycles équivalents",
        labels={"equivalent_cycles": "Cycles équivalents"},
    )


def pnl_comparison_chart(unhedged: pd.Series, hedged: pd.Series) -> go.Figure:
    frame = pd.concat([unhedged.rename("Avant hedge"), hedged.rename("Après hedge")], axis=1).melt(
        var_name="Portefeuille", value_name="P&L"
    )
    return px.histogram(
        frame,
        x="P&L",
        color="Portefeuille",
        barmode="overlay",
        marginal="box",
        opacity=0.6,
        title="Distribution du P&L avant / après hedge",
        labels={"P&L": "P&L (€)"},
    )


def pnl_boxplot(unhedged: pd.Series, hedged: pd.Series) -> go.Figure:
    frame = pd.concat([unhedged.rename("Avant"), hedged.rename("Après")], axis=1).melt(
        var_name="Hedge", value_name="P&L (€)"
    )
    return px.box(frame, x="Hedge", y="P&L (€)", points="all", title="Boxplots du P&L")


def pnl_waterfall(metrics: pd.Series | dict[str, float]) -> go.Figure:
    values = pd.Series(metrics)
    keys = [
        "discharge_revenue_eur",
        "charge_cost_eur",
        "variable_cost_eur",
        "degradation_cost_eur",
    ]
    amounts = [
        values.get(keys[0], 0),
        -values.get(keys[1], 0),
        -values.get(keys[2], 0),
        -values.get(keys[3], 0),
    ]
    return go.Figure(
        go.Waterfall(
            x=["Revenus décharge", "Coût charge", "Coûts variables", "Dégradation", "Net"],
            y=[*amounts, sum(amounts)],
            measure=["relative"] * 4 + ["total"],
        )
    ).update_layout(title="Waterfall du P&L", yaxis_title="Euros (€)")


def hedge_volumes_chart(volumes: pd.Series) -> go.Figure:
    frame = volumes.rename("Volume").reset_index()
    return px.bar(
        frame,
        x="index",
        y="Volume",
        title="Volumes de hedge par année et profil",
        labels={"index": "Produit", "Volume": "Volume signé (MW)"},
    )


def mean_variance_frontier(
    unhedged: pd.Series, unit_payoffs: pd.DataFrame, direction: pd.Series
) -> go.Figure:
    scales = np.linspace(0, 1.5, 31)
    rows = []
    for scale in scales:
        pnl = unhedged + unit_payoffs @ (direction * scale)
        rows.append(
            {
                "Niveau de hedge": scale,
                "Espérance (€)": pnl.mean(),
                "Écart-type (€)": pnl.std(ddof=0),
            }
        )
    return px.line(
        pd.DataFrame(rows),
        x="Écart-type (€)",
        y="Espérance (€)",
        markers=True,
        hover_data=["Niveau de hedge"],
        title="Frontière espérance / risque",
    )


def risk_table_chart(metrics: dict[str, float]) -> go.Figure:
    labels = list(metrics)
    values = [f"{metrics[key]:,.2f}" for key in labels]
    return go.Figure(
        data=[
            go.Table(
                header={"values": ["Indicateur", "Valeur"]}, cells={"values": [labels, values]}
            )
        ]
    ).update_layout(title="Principaux indicateurs de risque")
