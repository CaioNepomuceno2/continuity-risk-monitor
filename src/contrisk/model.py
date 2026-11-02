"""Modelos e baselines para P(violar o limite anual) e R$ de compensação até dezembro.

Validação por origem móvel: para prever o ano ``t``, os modelos são treinados só com anos ``< t``
(todos os meses de corte), e o ano ``t`` é avaliado em cada mês de corte.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from contrisk.features import feature_cols

PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 0.8,
          "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_estimators": 400,
          "verbose": -1, "n_jobs": 2, "deterministic": True, "force_row_wise": True}

# baselines de regra: escore = razão projeção / limite; decisão = escore > 1
BASELINES = {
    "extrapolacao_linear": "r_{ind}_proj_linear",
    "fracao_sazonal": "r_{ind}_proj_sazonal",
    "violou_ano_passado": "r_{ind}_anual_ant1",
}


def ano_inicio_treino(p: pd.DataFrame) -> int:
    return int(p.ano.min()) + 1  # o primeiro ano não tem fechamento anterior


def fit_classificador(tr: pd.DataFrame, alvo: str, seed: int = 0) -> lgb.LGBMClassifier:
    tr = tr[tr[alvo].notna()]
    m = lgb.LGBMClassifier(objective="binary", random_state=seed, **PARAMS)
    m.fit(tr[feature_cols(tr)], tr[alvo].astype(int), categorical_feature=["agente"])
    return m


def fit_compensacao(tr: pd.DataFrame, seed: int = 0) -> lgb.LGBMRegressor:
    tr = tr[tr.comp_resto_rs.notna()]
    m = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, random_state=seed, **PARAMS)
    m.fit(tr[feature_cols(tr)], tr.comp_resto_rs.clip(lower=0), categorical_feature=["agente"])
    return m


def _logit_baseline(tr: pd.DataFrame, te: pd.DataFrame, col: str, alvo: str) -> np.ndarray:
    """Converte o escore de regra em probabilidade (regressão logística por mês de corte, só no treino)."""
    out = np.full(len(te), np.nan)
    for m in te.mes_corte.unique():
        a = tr[(tr.mes_corte == m) & tr[col].notna() & tr[alvo].notna()]
        idx = np.where((te.mes_corte == m).values & te[col].notna().values)[0]
        if len(a) < 50 or len(idx) == 0:
            continue
        x = np.log(a[[col]].clip(lower=1e-3).values)
        lr = LogisticRegression().fit(x, a[alvo].astype(int))
        out[idx] = lr.predict_proba(np.log(te.iloc[idx][[col]].clip(lower=1e-3).values))[:, 1]
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Erro de calibração esperado (média ponderada de |observado - previsto| por faixa de probabilidade)."""
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = k == b
        if s.any():
            tot += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(tot)


def metricas(y: np.ndarray, escore: np.ndarray, decisao: np.ndarray, prob: np.ndarray | None,
             rs: np.ndarray) -> dict:
    """Discriminação, decisão binária, calibração e R$ dos conjuntos sinalizados."""
    ok = ~np.isnan(escore) & ~np.isnan(y)
    y, escore, decisao, rs = y[ok], escore[ok], decisao[ok].astype(bool), rs[ok]
    tp = (decisao & (y == 1)).sum()
    out = {
        "n": int(ok.sum()),
        "taxa_violacao": float(y.mean()),
        "auc": float(roc_auc_score(y, escore)) if 0 < y.mean() < 1 else np.nan,
        "precisao": float(tp / decisao.sum()) if decisao.sum() else np.nan,
        "recall": float(tp / (y == 1).sum()) if (y == 1).sum() else np.nan,
        "sinalizados": int(decisao.sum()),
        "rs_resto_sinalizados_frac": float(rs[decisao].sum() / rs.sum()) if rs.sum() else np.nan,
    }
    k = max(1, round(0.1 * len(escore)))
    top = np.argsort(-escore)[:k]
    out["recall_top10"] = float(y[top].sum() / y.sum()) if y.sum() else np.nan
    if prob is not None:
        pr = prob[ok]
        out["brier"] = float(brier_score_loss(y, pr))
        out["ece"] = ece(y, pr)
    return out


def backtest(p: pd.DataFrame, anos_teste: list[int], cortes_relatorio: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (métricas por ano x corte x indicador x método, previsões do modelo no teste)."""
    linhas, prevs = [], []
    for t in anos_teste:
        tr = p[(p.ano < t) & (p.ano >= ano_inicio_treino(p))]
        te = p[(p.ano == t) & p.mes_corte.isin(cortes_relatorio)].copy()
        comp_model = fit_compensacao(tr)
        te["pred_comp_resto_rs"] = comp_model.predict(te[feature_cols(te)])
        for ind in ("dec", "fec"):
            alvo = f"viol_{ind}"
            clf = fit_classificador(tr, alvo)
            te[f"p_{alvo}"] = clf.predict_proba(te[feature_cols(te)])[:, 1]
            base_probs = {nome: _logit_baseline(tr, te, col.format(ind=ind), alvo)
                          for nome, col in BASELINES.items()}
            for m in cortes_relatorio:
                s = te[te.mes_corte == m]
                sel = (te.mes_corte == m).values
                y = s[alvo].values.astype(float)
                rs = s.comp_resto_rs.fillna(0).values
                pm = s[f"p_{alvo}"].values
                linhas.append({"ano": t, "mes_corte": m, "indicador": ind.upper(), "metodo": "modelo",
                               **metricas(y, pm, pm >= 0.5, pm, rs)})
                for nome, col in BASELINES.items():
                    esc = s[col.format(ind=ind)].values.astype(float)
                    linhas.append({"ano": t, "mes_corte": m, "indicador": ind.upper(), "metodo": nome,
                                   **metricas(y, esc, esc > 1, base_probs[nome][sel], rs)})
        prevs.append(te[["agente", "conjunto_id", "conjunto", "ano", "mes_corte", "viol_dec", "viol_fec",
                         "p_viol_dec", "p_viol_fec", "comp_resto_rs", "pred_comp_resto_rs",
                         "compensacao_rs_acum", "comp_anual_rs_ant1", "hist_comp_resto"]])
    return pd.DataFrame(linhas), pd.concat(prevs, ignore_index=True)


def metricas_compensacao(prev: pd.DataFrame) -> pd.DataFrame:
    """Erro do R$ de compensação de m+1 a dezembro: modelo vs extrapolação e vs histórico."""
    d = prev.copy()
    d["pred_linear"] = d.compensacao_rs_acum * (12 - d.mes_corte) / d.mes_corte
    d["pred_historico"] = d.hist_comp_resto
    linhas = []
    for (ano, m), g in d.groupby(["ano", "mes_corte"]):
        y = g.comp_resto_rs.clip(lower=0)
        for nome, col in [("modelo", "pred_comp_resto_rs"), ("extrapolacao_linear", "pred_linear"),
                          ("historico_5a", "pred_historico")]:
            x = g[col]
            ok = x.notna() & y.notna()
            linhas.append({"ano": ano, "mes_corte": m, "metodo": nome,
                           "mae_rs": float((x[ok] - y[ok]).abs().mean()),
                           "erro_total_pct": float(100 * (x[ok].sum() / y[ok].sum() - 1)),
                           "cobertura": float(ok.mean())})
    return pd.DataFrame(linhas)


def prever(p: pd.DataFrame, ano: int, seeds: int = 3) -> pd.DataFrame:
    """Treina com todos os anos fechados antes de ``ano`` e prevê cada conjunto no último mês disponível."""
    tr = p[(p.ano < ano) & (p.ano >= ano_inicio_treino(p))]
    te = p[p.ano == ano]
    te = te[te.mes_corte == te.groupby("conjunto_id").mes_corte.transform("max")].copy()
    X = te[feature_cols(te)]
    for alvo in ("viol_dec", "viol_fec"):
        te[f"p_{alvo}"] = np.mean([fit_classificador(tr, alvo, s).predict_proba(X)[:, 1] for s in range(seeds)], 0)
    te["pred_comp_resto_rs"] = np.mean([fit_compensacao(tr, s).predict(X) for s in range(seeds)], 0)
    te["p_viol_qualquer"] = 1 - (1 - te.p_viol_dec) * (1 - te.p_viol_fec)  # aproximação (independência)
    return te
