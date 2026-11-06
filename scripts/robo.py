"""Robô mensal: baixa os dados da ANEEL, recalcula o risco do ano corrente e publica os relatórios.

    python scripts/robo.py                 # download + previsão do ano corrente + relatórios + Power BI
    python scripts/robo.py --sem-download  # usa os arquivos já baixados em data/raw
    python scripts/robo.py --backtest      # também refaz o backtest 2021-2025 (mais lento)
"""
import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

from contrisk import build
from contrisk import features as F
from contrisk import model as M
from contrisk import report as R
from contrisk.download import download
from contrisk.snapshot import preparar_sem_falhar

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM, REPORTS = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "reports"
SNAPSHOTS, MANIFESTO = ROOT / "data" / "snapshots", ROOT / "snapshots" / "manifesto.csv"
FIG, PBI = REPORTS / "figures", ROOT / "powerbi"
ANOS_TESTE, CORTES_RELATORIO = [2021, 2022, 2023, 2024, 2025], [3, 6, 9]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("robo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-download", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    for d in (INTERIM, REPORTS, FIG):
        d.mkdir(parents=True, exist_ok=True)

    if not args.sem_download:
        download(RAW, atualizar=True)
        # cópia datada do que a ANEEL publicou; nunca derruba o robô (a publicação é um passo à parte do workflow)
        preparar_sem_falhar(RAW, SNAPSHOTS, MANIFESTO)
    base = build.base_mensal(RAW)
    painel = F.painel(base)
    ano = int(base.ano.max())
    log.info("base: %d linhas; painel: %d linhas; ano corrente: %d", len(base), len(painel), ano)

    hist = R.historico(base)
    hist.to_csv(REPORTS / "historico_violacoes.csv", index=False)
    R.compensacao_vs_violacao(base, range(2021, 2026)).to_csv(REPORTS / "compensacao_vs_violacao.csv", index=False)
    R.fig_historico(hist, FIG)

    if args.backtest:
        res, prev_bt = M.backtest(painel, ANOS_TESTE, CORTES_RELATORIO)
        res.to_csv(REPORTS / "backtest_por_ano.csv", index=False)
        med = R.resumo_backtest(res)
        med.to_csv(REPORTS / "backtest_metricas.csv", index=False)
        M.metricas_compensacao(prev_bt).to_csv(REPORTS / "backtest_compensacao.csv", index=False)
        prev_bt.to_parquet(INTERIM / "bt_prev.parquet", index=False)
        R.fig_backtest(med, FIG)
        R.fig_calibracao(prev_bt, FIG)
        log.info("backtest:\n%s", med.round(3).to_string())

    prev = M.prever(painel, ano)
    lista = R.lista_conjuntos(prev)
    rank = R.ranking_distribuidoras(prev)
    lista.to_csv(REPORTS / f"previsao_{ano}.csv", index=False)
    rank.to_csv(REPORTS / f"ranking_distribuidoras_{ano}.csv", index=False)
    lista[lista.agente.isin(R.DESTAQUE)].to_csv(REPORTS / f"energisa_ms_mt_{ano}.csv", index=False)
    R.fig_ranking(rank, FIG, ano)
    R.fig_destaque(lista, FIG, ano)
    gerado = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (REPORTS / "resumo.md").write_text(R.resumo_md(lista, rank, ano, gerado), encoding="utf-8")
    R.salvar_json({"ano": ano, "gerado_em": gerado, "conjuntos": len(lista),
                   "mes_dados_min": int(lista.mes_corte.min()), "mes_dados_max": int(lista.mes_corte.max()),
                   "violacoes_dec_esperadas": lista.p_viol_dec.sum(),
                   "violacoes_fec_esperadas": lista.p_viol_fec.sum(),
                   "conjuntos_risco_alto": int((lista.p_viol_qualquer >= 0.5).sum()),
                   "compensacao_paga_rs": lista.compensacao_rs_acum.sum(),
                   "compensacao_esperada_resto_rs": lista.pred_comp_resto_rs.sum()},
                  REPORTS / "status.json")
    R.powerbi(base, lista, PBI)
    log.info("relatórios em %s; modelo estrela em %s", REPORTS, PBI)


if __name__ == "__main__":
    main()
