"""Painel conjunto x ano x mês de corte.

Para o ano ``y`` e o mês de corte ``m``, as features usam só:

* os meses ``1..m`` do próprio ano ``y`` (acumulado, últimos 3 meses, compensações pagas);
* os anos ``< y`` completos (sazonalidade típica, fechamento e violações anteriores);
* o limite anual de ``y``, que é publicado antes do início do ano.

Os alvos usam o ano ``y`` completo: violação do limite anual de DEC/FEC e R$ de compensação pagos de
``m+1`` a dezembro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CORTES = list(range(1, 12))
ALVOS = ["viol_dec", "viol_fec", "dec_anual", "fec_anual", "comp_resto_rs"]
CATEG = ["agente"]


def anual(base: pd.DataFrame) -> pd.DataFrame:
    """Fechamento anual por conjunto (só anos com os 12 meses)."""
    g = base.groupby(["conjunto_id", "ano"])
    a = g.agg(n_meses=("mes", "nunique"), dec_anual=("dec", "sum"), fec_anual=("fec", "sum"),
              comp_anual_rs=("compensacao_rs", "sum"), lim_dec=("lim_dec", "first"),
              lim_fec=("lim_fec", "first")).reset_index()
    a = a[a.n_meses == 12].drop(columns="n_meses")
    a["viol_dec"] = (a.dec_anual > a.lim_dec).astype(float)
    a["viol_fec"] = (a.fec_anual > a.lim_fec).astype(float)
    a.loc[a.lim_dec.isna() | (a.lim_dec <= 0), "viol_dec"] = np.nan
    a.loc[a.lim_fec.isna() | (a.lim_fec <= 0), "viol_fec"] = np.nan
    return a


def _acumulados(base: pd.DataFrame) -> pd.DataFrame:
    """Acumulados até cada mês, por conjunto x ano (inclui anos incompletos)."""
    b = base.sort_values(["conjunto_id", "ano", "mes"]).copy()
    g = b.groupby(["conjunto_id", "ano"])
    for v in ["dec", "fec", "compensacao_rs"]:
        b[f"{v}_acum"] = g[v].cumsum()
        b[f"{v}_ult3"] = g[v].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return b


def _perfil_historico(acum: pd.DataFrame, fech: pd.DataFrame, anos_hist: int = 5) -> pd.DataFrame:
    """Para cada conjunto x ano y x mês m: médias dos anos y-anos_hist..y-1 de

    * fração do DEC/FEC anual realizada até m;
    * DEC/FEC realizados de m+1 a dezembro;
    * compensações pagas de m+1 a dezembro.
    """
    x = acum.merge(fech[["conjunto_id", "ano", "dec_anual", "fec_anual", "comp_anual_rs"]],
                   on=["conjunto_id", "ano"], how="inner")
    x["frac_dec"] = x.dec_acum / x.dec_anual.replace(0, np.nan)
    x["frac_fec"] = x.fec_acum / x.fec_anual.replace(0, np.nan)
    x["dec_resto"] = x.dec_anual - x.dec_acum
    x["fec_resto"] = x.fec_anual - x.fec_acum
    x["comp_resto"] = x.comp_anual_rs - x.compensacao_rs_acum
    cols = ["frac_dec", "frac_fec", "dec_resto", "fec_resto", "comp_resto"]
    x = x[["conjunto_id", "ano", "mes"] + cols]
    anos = sorted(x.ano.unique())
    saidas = []
    for y in range(min(anos) + 1, max(anos) + 2):
        h = x[(x.ano < y) & (x.ano >= y - anos_hist)]
        if h.empty:
            continue
        m = h.groupby(["conjunto_id", "mes"])[cols].mean().add_prefix("hist_")
        m["hist_n_anos"] = h.groupby(["conjunto_id", "mes"]).ano.nunique()
        saidas.append(m.reset_index().assign(ano=y))
    return pd.concat(saidas, ignore_index=True)


def painel(base: pd.DataFrame, cortes: list[int] = CORTES, anos_hist: int = 5) -> pd.DataFrame:
    """Uma linha por conjunto x ano x mês de corte, com features e (quando conhecidos) alvos."""
    fech = anual(base)
    acum = _acumulados(base)
    perfil = _perfil_historico(acum, fech, anos_hist)

    p = acum[acum.mes.isin(cortes)].rename(columns={"mes": "mes_corte"})
    p = p[["agente", "conjunto_id", "conjunto", "ano", "mes_corte", "numcon", "lim_dec", "lim_fec",
           "dec_acum", "fec_acum", "compensacao_rs_acum", "dec_ult3", "fec_ult3", "compensacao_rs_ult3"]]
    p = p.merge(perfil.rename(columns={"mes": "mes_corte"}), on=["conjunto_id", "ano", "mes_corte"], how="left")

    # fechamentos dos anos anteriores
    ant = fech[["conjunto_id", "ano", "dec_anual", "fec_anual", "comp_anual_rs", "lim_dec", "lim_fec",
                "viol_dec", "viol_fec"]]
    for k in (1, 2):
        a = ant.copy()
        a["ano"] = a.ano + k
        a = a.rename(columns={c: f"{c}_ant{k}" for c in a.columns if c not in ("conjunto_id", "ano")})
        p = p.merge(a, on=["conjunto_id", "ano"], how="left")

    # alvos do ano y
    alvo = fech[["conjunto_id", "ano", "dec_anual", "fec_anual", "comp_anual_rs", "viol_dec", "viol_fec"]]
    p = p.merge(alvo, on=["conjunto_id", "ano"], how="left")
    p["comp_resto_rs"] = p.comp_anual_rs - p.compensacao_rs_acum
    p = p.drop(columns="comp_anual_rs")

    # features derivadas (projeções que um analista faria à mão)
    m = p.mes_corte
    p["dec_proj_linear"] = p.dec_acum * 12 / m
    p["fec_proj_linear"] = p.fec_acum * 12 / m
    p["dec_proj_sazonal"] = p.dec_acum / p.hist_frac_dec.clip(lower=0.05)
    p["fec_proj_sazonal"] = p.fec_acum / p.hist_frac_fec.clip(lower=0.05)
    p["dec_proj_resto"] = p.dec_acum + p.hist_dec_resto
    p["fec_proj_resto"] = p.fec_acum + p.hist_fec_resto
    for ind in ("dec", "fec"):
        lim = p[f"lim_{ind}"].where(p[f"lim_{ind}"] > 0)
        for c in (f"{ind}_acum", f"{ind}_proj_linear", f"{ind}_proj_sazonal", f"{ind}_proj_resto",
                  f"{ind}_anual_ant1", f"{ind}_anual_ant2"):
            p[f"r_{c}"] = p[c] / lim
        p[f"folga_{ind}"] = lim - p[f"{ind}_acum"]
    p["lim_dec_var"] = p.lim_dec / p.lim_dec_ant1
    p["agente"] = p.agente.astype("category")
    return p.reset_index(drop=True)


def feature_cols(p: pd.DataFrame) -> list[str]:
    fora = set(ALVOS) | {"conjunto_id", "conjunto", "ano"}
    return [c for c in p.columns if c not in fora and not c.startswith(("p_", "pred_"))]
