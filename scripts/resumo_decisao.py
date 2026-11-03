"""O que a lista de alertas significa para quem decide (backtest 2021-2025, DEC).

Para cada método e mês de corte, por ano, conta:

* conjuntos sinalizados, acertos (violaram de fato) e alarmes falsos;
* violações que o método deixou passar;
* sinalizados que **ainda estavam dentro do limite** no mês de corte (onde ainda dá para agir) e, entre eles,
  quantos de fato violaram;
* R$ de compensação pago de m+1 a dezembro nos conjuntos sinalizados.

E grava a média por ano em ``reports/resumo_decisao.csv``.

Pré-requisito: ``python scripts/robo.py --backtest`` (gera ``data/interim/painel.parquet`` e
``data/interim/bt_prev.parquet``). Não baixa nada.
"""
import numpy as np
import pandas as pd

from contrisk.model import BASELINES

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "reports"

cols = ["conjunto_id", "ano", "mes_corte", "viol_dec", "comp_resto_rs", "folga_dec"] + [
    c.format(ind="dec") for c in BASELINES.values()]
painel = pd.read_parquet(INTERIM / "painel.parquet", columns=cols)
prev = pd.read_parquet(INTERIM / "bt_prev.parquet", columns=["conjunto_id", "ano", "mes_corte", "p_viol_dec"])
d = prev.merge(painel, on=["conjunto_id", "ano", "mes_corte"], how="left")
d = d[d.viol_dec.notna()]

metodos = {"modelo": ("p_viol_dec", 0.5, ">=")} | {
    nome: (col.format(ind="dec"), 1.0, ">") for nome, col in BASELINES.items()}

linhas = []
for (ano, m), g in d.groupby(["ano", "mes_corte"]):
    for nome, (col, lim, op) in metodos.items():
        s = g[g[col].notna()]
        flag = (s[col] >= lim) if op == ">=" else (s[col] > lim)
        y = s.viol_dec == 1
        dentro = s.folga_dec > 0
        rs = s.comp_resto_rs.clip(lower=0).fillna(0)
        linhas.append({
            "ano": ano, "mes_corte": m, "metodo": nome,
            "conjuntos": len(s), "violacoes": int(y.sum()),
            "sinalizados": int(flag.sum()), "acertos": int((flag & y).sum()),
            "alarmes_falsos": int((flag & ~y).sum()), "violacoes_perdidas": int((~flag & y).sum()),
            "sinalizados_dentro_do_limite": int((flag & dentro).sum()),
            "acertos_dentro_do_limite": int((flag & dentro & y).sum()),
            "rs_mi_resto_sinalizados": float(rs[flag].sum() / 1e6),
            "rs_mi_resto_total": float(rs.sum() / 1e6),
        })
por_ano = pd.DataFrame(linhas)
media = por_ano.drop(columns="ano").groupby(["mes_corte", "metodo"]).mean().reset_index()
media["precisao"] = media.acertos / media.sinalizados
media["recall"] = media.acertos / media.violacoes
media["rs_mil_por_sinalizado"] = media.rs_mi_resto_sinalizados * 1e3 / media.sinalizados
media = media.round(4)
media.to_csv(OUT / "resumo_decisao.csv", index=False)
por_ano.to_csv(OUT / "resumo_decisao_por_ano.csv", index=False)
with pd.option_context("display.width", 200):
    print(media[media.metodo.isin(["modelo", "extrapolacao_linear"])].to_string(index=False))
assert np.isfinite(media.precisao).all()
