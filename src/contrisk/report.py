"""Relatórios, figuras e modelo estrela para Power BI."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contrisk.features import anual

DESTAQUE = ["EMS", "EMT"]
NOMES_METODO = {"modelo": "Modelo (LightGBM)", "extrapolacao_linear": "Extrapolação linear",
                "fracao_sazonal": "Fração sazonal histórica", "violou_ano_passado": "Violou no ano passado"}
CORES = {"modelo": "#2a78d6", "extrapolacao_linear": "#ec6935", "fracao_sazonal": "#1baf7a",
         "violou_ano_passado": "#9a8fbf"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "axes.axisbelow": True,
                     "font.size": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _brl(v: float) -> str:
    return f"R$ {v / 1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")


def _num(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


# --------------------------------------------------------------------------- tabelas
def historico(base: pd.DataFrame) -> pd.DataFrame:
    a = anual(base)
    a = a[a.viol_dec.notna()]
    return a.groupby("ano").agg(conjuntos=("conjunto_id", "size"), frac_viol_dec=("viol_dec", "mean"),
                                frac_viol_fec=("viol_fec", "mean"),
                                compensacao_rs=("comp_anual_rs", "sum")).reset_index()


def compensacao_vs_violacao(base: pd.DataFrame, anos: range) -> pd.DataFrame:
    """Relação empírica: R$ pagos por conjunto conforme a razão DEC anual / limite."""
    a = anual(base)
    a = a[a.ano.isin(list(anos)) & a.viol_dec.notna()].copy()
    a["faixa_dec_limite"] = pd.cut(a.dec_anual / a.lim_dec, [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2, np.inf],
                                   labels=["<0,5", "0,5-0,8", "0,8-1,0", "1,0-1,2", "1,2-1,5", "1,5-2", ">2"])
    g = a.groupby("faixa_dec_limite", observed=False)
    out = g.agg(conjunto_anos=("conjunto_id", "size"), comp_media_rs=("comp_anual_rs", "mean"),
                comp_total_rs=("comp_anual_rs", "sum")).reset_index()
    out["frac_comp_total"] = out.comp_total_rs / out.comp_total_rs.sum()
    return out


def resumo_backtest(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["taxa_violacao", "auc", "brier", "ece", "precisao", "recall", "rs_resto_sinalizados_frac"]
    return res.groupby(["indicador", "mes_corte", "metodo"])[cols].mean().reset_index()


def ranking_distribuidoras(prev: pd.DataFrame) -> pd.DataFrame:
    g = prev.groupby("agente", observed=True)
    r = g.agg(conjuntos=("conjunto_id", "size"), mes_dados=("mes_corte", "max"),
              violacoes_dec_esperadas=("p_viol_dec", "sum"), violacoes_fec_esperadas=("p_viol_fec", "sum"),
              conjuntos_risco_alto=("p_viol_qualquer", lambda s: int((s >= 0.5).sum())),
              compensacao_paga_rs=("compensacao_rs_acum", "sum"),
              compensacao_esperada_resto_rs=("pred_comp_resto_rs", "sum")).reset_index()
    r["frac_conjuntos_risco_alto"] = r.conjuntos_risco_alto / r.conjuntos
    return r.sort_values("compensacao_esperada_resto_rs", ascending=False, ignore_index=True)


def lista_conjuntos(prev: pd.DataFrame) -> pd.DataFrame:
    cols = ["agente", "conjunto_id", "conjunto", "ano", "mes_corte", "numcon", "dec_acum", "lim_dec",
            "dec_proj_linear", "p_viol_dec", "fec_acum", "lim_fec", "fec_proj_linear", "p_viol_fec",
            "p_viol_qualquer", "compensacao_rs_acum", "pred_comp_resto_rs"]
    out = prev[cols].copy()
    out["folga_dec"] = out.lim_dec - out.dec_acum
    out["folga_fec"] = out.lim_fec - out.fec_acum
    out["agente"] = out.agente.astype(str)
    return out.sort_values(["p_viol_qualquer", "pred_comp_resto_rs"], ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- figuras
def fig_backtest(res_med: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ind in zip(axes, ["DEC", "FEC"], strict=True):
        d = res_med[res_med.indicador == ind]
        for met, g in d.groupby("metodo"):
            ax.plot(g.mes_corte, g.brier, marker="o", lw=2.2 if met == "modelo" else 1.6,
                    color=CORES[met], label=NOMES_METODO[met])
        ax.set_xticks([3, 6, 9], ["março", "junho", "setembro"])
        ax.set_title(f"Violação do limite anual de {ind}\nBrier score (menor é melhor)")
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Backtest 2021-2025: previsão do fechamento do ano a partir dos dados até o mês de corte",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "01_backtest_brier.png")
    plt.close(fig)


def fig_calibracao(prev_bt: pd.DataFrame, out: Path, mes: int = 6) -> None:
    d = prev_bt[(prev_bt.mes_corte == mes) & prev_bt.viol_dec.notna()]
    bins = np.linspace(0, 1, 11)
    k = np.clip(np.digitize(d.p_viol_dec, bins[1:-1]), 0, 9)
    g = d.groupby(k).agg(prev=("p_viol_dec", "mean"), obs=("viol_dec", "mean"), n=("viol_dec", "size"))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax.scatter(g.prev, g.obs, s=np.sqrt(g.n) * 4, color=CORES["modelo"], zorder=3)
    ax.plot(g.prev, g.obs, color=CORES["modelo"])
    ax.set_xlabel("probabilidade prevista em junho")
    ax.set_ylabel("fração que de fato violou o DEC no ano")
    ax.set_title("Calibração da P(violar DEC) em junho\n(backtest 2021-2025)")
    fig.tight_layout()
    fig.savefig(out / "02_calibracao_junho.png")
    plt.close(fig)


def fig_historico(hist: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(hist.ano, hist.frac_viol_dec * 100, marker="o", color=CORES["modelo"], label="DEC acima do limite")
    ax.plot(hist.ano, hist.frac_viol_fec * 100, marker="o", color=CORES["extrapolacao_linear"],
            label="FEC acima do limite")
    ax.set_ylabel("% dos conjuntos")
    ax2 = ax.twinx()
    ax2.bar(hist.ano, hist.compensacao_rs / 1e9, color="#d9d9d9", zorder=0, width=0.6)
    ax2.set_ylabel("compensações pagas (R$ bi)")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Menos conjuntos violam os limites, mas as compensações pagas cresceram")
    fig.tight_layout()
    fig.savefig(out / "03_historico.png")
    plt.close(fig)


def fig_ranking(rank: pd.DataFrame, out: Path, ano: int, n: int = 15) -> None:
    d = rank.head(n).iloc[::-1]
    cores = ["#ec6935" if a in DESTAQUE else CORES["modelo"] for a in d.agente]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(d.agente.astype(str), d.compensacao_esperada_resto_rs / 1e6, color=cores)
    for y, (v, c) in enumerate(zip(d.compensacao_esperada_resto_rs, d.conjuntos_risco_alto, strict=True)):
        ax.text(v / 1e6, y, f"  {c} conj. risco alto", va="center", fontsize=8, color="#555")
    ax.set_xlabel("compensação esperada até dezembro (R$ mi)")
    ax.set_title(f"{ano}: distribuidoras com mais compensação esperada até o fim do ano")
    fig.tight_layout()
    fig.savefig(out / "04_ranking_distribuidoras.png")
    plt.close(fig)


def fig_destaque(lista: pd.DataFrame, out: Path, ano: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, ag in zip(axes, DESTAQUE, strict=True):
        d = lista[lista.agente == ag]
        if d.empty:
            continue
        sc = ax.scatter(d.dec_acum / d.lim_dec, d.p_viol_dec, s=np.sqrt(d.numcon.fillna(0)) / 3,
                        c=d.pred_comp_resto_rs / 1e3, cmap="viridis", alpha=0.85)
        ax.axhline(0.5, color="#999", lw=1, ls="--")
        ax.margins(x=0.15)
        ax.set_xlabel("DEC acumulado / limite anual")
        ax.set_title(f"{ag}: {len(d)} conjuntos (dados até o mês {int(d.mes_corte.max())})")
        for _, r in d.nlargest(4, "p_viol_dec").iterrows():
            ax.annotate(str(r.conjunto)[:18], (r.dec_acum / r.lim_dec, r.p_viol_dec), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        fig.colorbar(sc, ax=ax, label="compensação esperada até dez. (R$ mil)")
    axes[0].set_ylabel("P(violar o DEC no ano)")
    fig.suptitle(f"{ano}: risco por conjunto na Energisa MS e MT", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "05_energisa_ms_mt.png")
    plt.close(fig)


# --------------------------------------------------------------------------- resumo e Power BI
def resumo_md(lista: pd.DataFrame, rank: pd.DataFrame, ano: int, gerado_em: str) -> str:
    alto = lista[lista.p_viol_qualquer >= 0.5]
    linhas = [f"# Monitor de risco regulatório de continuidade: {ano}", "",
              (f"Gerado em {gerado_em} a partir dos dados abertos da ANEEL. Dados de cada conjunto até o "
              f"último mês publicado (mês {int(lista.mes_corte.min())} a {int(lista.mes_corte.max())})."), "",
              f"* Conjuntos monitorados: **{len(lista)}**",
              (f"* Violações de DEC esperadas no ano: **{lista.p_viol_dec.sum():.0f}**; de FEC: "
              f"**{lista.p_viol_fec.sum():.0f}**"),
              f"* Conjuntos com P(violar DEC ou FEC) ≥ 50%: **{len(alto)}**",
              (f"* Compensações já pagas no ano: **{_brl(lista.compensacao_rs_acum.sum())}**; esperadas até "
              f"dezembro: **{_brl(lista.pred_comp_resto_rs.sum())}**"), "",
              "## Energisa MS e MT", ""]
    for ag in DESTAQUE:
        d = lista[lista.agente == ag]
        if d.empty:
            continue
        linhas += [(f"**{ag}**: {len(d)} conjuntos, {int((d.p_viol_qualquer >= 0.5).sum())} com risco alto, "
                   f"compensação esperada até dezembro de {_brl(d.pred_comp_resto_rs.sum())}."), "",
                   "| Conjunto | DEC acum. / limite | P(DEC) | FEC acum. / limite | P(FEC) | R$ esperado até dez. |",
                   "|---|---|---|---|---|---|"]
        for _, r in d.head(10).iterrows():
            linhas.append(f"| {r.conjunto} | {_num(r.dec_acum)} / {_num(r.lim_dec, 0)} | {r.p_viol_dec:.0%} | "
                          f"{_num(r.fec_acum)} / {_num(r.lim_fec, 0)} | {r.p_viol_fec:.0%} | "
                          f"{_brl(r.pred_comp_resto_rs)} |")
        linhas.append("")
    linhas += ["## Distribuidoras com mais compensação esperada até dezembro", "",
               "| Distribuidora | Conjuntos | Risco alto | Viol. DEC esperadas | R$ esperado até dez. |",
               "|---|---|---|---|---|"]
    for _, r in rank.head(15).iterrows():
        linhas.append(f"| {r.agente} | {r.conjuntos} | {r.conjuntos_risco_alto} | "
                      f"{r.violacoes_dec_esperadas:.0f} | {_brl(r.compensacao_esperada_resto_rs)} |")
    linhas += ["", ("Probabilidades e valores são estimativas de um modelo estatístico; ver o README do "
               "projeto para método, backtest e limitações.")]
    return "\n".join(linhas) + "\n"


def powerbi(base: pd.DataFrame, lista: pd.DataFrame, out: Path, ano_min: int = 2020) -> None:
    """Modelo estrela em Parquet: fatos mensais e de previsão, dimensões de conjunto, distribuidora e tempo."""
    out.mkdir(parents=True, exist_ok=True)
    b = base[base.ano >= ano_min].copy()
    b["data_id"] = b.ano * 100 + b.mes
    b[["conjunto_id", "data_id", "dec", "fec", "numcon", "compensacao_rs"]].to_parquet(
        out / "fato_indicador_mensal.parquet", index=False)
    f = lista.copy()
    f["data_id"] = f.ano * 100 + f.mes_corte
    f.drop(columns=["agente", "conjunto"]).to_parquet(out / "fato_previsao.parquet", index=False)
    ult = b.sort_values(["ano", "mes"]).groupby("conjunto_id").last().reset_index()
    ult[["conjunto_id", "conjunto", "agente", "numcon", "lim_dec", "lim_fec"]].rename(
        columns={"numcon": "consumidores_ultimo_mes", "lim_dec": "lim_dec_ano_atual",
                 "lim_fec": "lim_fec_ano_atual"}).to_parquet(out / "dim_conjunto.parquet", index=False)
    pd.DataFrame({"agente": sorted(b.agente.dropna().unique())}).assign(
        destaque=lambda d: d.agente.isin(DESTAQUE)).to_parquet(out / "dim_distribuidora.parquet", index=False)
    t = b[["ano", "mes"]].drop_duplicates().sort_values(["ano", "mes"])
    t["data_id"] = t.ano * 100 + t.mes
    t["data"] = pd.to_datetime({"year": t.ano, "month": t.mes, "day": 1})
    t["trimestre"] = (t.mes - 1) // 3 + 1
    t.to_parquet(out / "dim_tempo.parquet", index=False)


def salvar_json(obj: dict, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")
