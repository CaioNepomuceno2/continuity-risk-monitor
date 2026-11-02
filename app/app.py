"""Painel do monitor de risco regulatório de continuidade.

    pip install -e ".[app]"
    streamlit run app/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
PBI = ROOT / "powerbi"

st.set_page_config(page_title="Monitor de risco DEC/FEC", layout="wide")
status = json.loads((REP / "status.json").read_text(encoding="utf-8"))
ano = status["ano"]
lista = pd.read_csv(REP / f"previsao_{ano}.csv")
rank = pd.read_csv(REP / f"ranking_distribuidoras_{ano}.csv")
mensal = pd.read_parquet(PBI / "fato_indicador_mensal.parquet")


def brl(v: float) -> str:
    return f"R$ {v / 1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")


st.title(f"Risco de violar os limites de DEC/FEC em {ano}")
st.caption(f"Dados abertos da ANEEL; atualizado em {status['gerado_em']}. "
           "Probabilidades e valores são estimativas de um modelo; veja o README para o backtest.")

agentes = ["Todas"] + sorted(lista.agente.unique())
padrao = agentes.index("EMS") if "EMS" in agentes else 0
ag = st.sidebar.selectbox("Distribuidora", agentes, index=padrao)
limiar = st.sidebar.slider("Risco alto a partir de", 0.1, 0.9, 0.5, 0.05)
d = lista if ag == "Todas" else lista[lista.agente == ag]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Conjuntos", len(d))
c2.metric("Com risco alto", int((d.p_viol_qualquer >= limiar).sum()))
c3.metric("Compensação já paga", brl(d.compensacao_rs_acum.sum()))
c4.metric("Compensação esperada até dez.", brl(d.pred_comp_resto_rs.sum()))

if ag == "Todas":
    st.subheader("Distribuidoras")
    fig = px.bar(rank.head(25), x="compensacao_esperada_resto_rs", y="agente", orientation="h",
                 hover_data=["conjuntos", "conjuntos_risco_alto", "violacoes_dec_esperadas"],
                 labels={"compensacao_esperada_resto_rs": "R$ esperado até dezembro", "agente": ""})
    fig.update_layout(yaxis={"autorange": "reversed"}, height=600)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Conjuntos priorizados")
tab = d.assign(dec_sobre_limite=d.dec_acum / d.lim_dec, fec_sobre_limite=d.fec_acum / d.lim_fec)
st.dataframe(
    tab[["agente", "conjunto", "mes_corte", "dec_acum", "lim_dec", "p_viol_dec", "fec_acum", "lim_fec",
         "p_viol_fec", "compensacao_rs_acum", "pred_comp_resto_rs"]],
    use_container_width=True, hide_index=True,
    column_config={
        "p_viol_dec": st.column_config.ProgressColumn("P(DEC)", min_value=0, max_value=1, format="%.0f%%"),
        "p_viol_fec": st.column_config.ProgressColumn("P(FEC)", min_value=0, max_value=1, format="%.0f%%"),
        "pred_comp_resto_rs": st.column_config.NumberColumn("R$ esperado até dez.", format="%.0f"),
        "compensacao_rs_acum": st.column_config.NumberColumn("R$ pago no ano", format="%.0f"),
        "mes_corte": "Dados até o mês",
    },
)

st.subheader("Detalhe do conjunto")
nome = st.selectbox("Conjunto", d.conjunto.tolist())
linha = d[d.conjunto == nome].iloc[0]
serie = mensal[mensal.conjunto_id == linha.conjunto_id].copy()
serie["ano"] = serie.data_id // 100
serie["mes"] = serie.data_id % 100
serie["dec_acum"] = serie.groupby("ano").dec.cumsum()
fig = px.line(serie, x="mes", y="dec_acum", color=serie.ano.astype(str),
              labels={"mes": "mês", "dec_acum": "DEC acumulado (h)", "color": "ano"})
fig.add_hline(y=linha.lim_dec, line_dash="dash", annotation_text=f"limite {ano}")
st.plotly_chart(fig, use_container_width=True)
st.write(f"P(violar DEC) = **{linha.p_viol_dec:.0%}**, P(violar FEC) = **{linha.p_viol_fec:.0%}**, "
         f"compensação esperada até dezembro: **{brl(linha.pred_comp_resto_rs)}**.")
