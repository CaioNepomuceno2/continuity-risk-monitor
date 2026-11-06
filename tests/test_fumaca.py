"""Teste de fumaça do núcleo: arquivos no formato da ANEEL -> base mensal -> painel -> probabilidades -> R$.

Gera, em disco temporário, um conjunto sintético minúsculo (40 conjuntos, 2018 a 2023, o último ano parado em
julho) com os mesmos nomes de arquivo e colunas que o robô lê da ANEEL, e roda o caminho real do robô
(``build.base_mensal``, ``features.painel``, ``model.prever``, ``report.lista_conjuntos``). Sem rede, sem
dados reais, com sementes fixas.
"""
import numpy as np
import pandas as pd
import pytest
import requests

from contrisk import build
from contrisk import features as F
from contrisk import model as M
from contrisk import report as R

ANOS = list(range(2018, 2024))
ANO_ATUAL, MESES_PUBLICADOS = 2023, 7
N_CONJ = 40
LIM_DEC, LIM_FEC = 12.0, 7.0
SAZON = 1 + 0.4 * np.sin(2 * np.pi * (np.arange(12)) / 12)   # média 1: o ano soma 12 x o nível do conjunto


def _gerar(raw, seed=7):
    """Escreve os três arquivos da ANEEL em ``raw`` e devolve a "verdade" do ano corrente (12 meses)."""
    rng = np.random.default_rng(seed)
    ind, comp, lim, verdade = [], [], [], []
    for k in range(N_CONJ):
        cid, agente = 1000 + k, ("AAA" if k % 2 == 0 else "BBB")
        nivel = rng.uniform(0.3, 1.6)               # nível persistente do conjunto (1,0 = no limite de DEC)
        numcon = 1000.0 * rng.uniform(1, 5)
        for ano in ANOS:
            n_ano = nivel * rng.lognormal(0, 0.1)
            dec = n_ano * SAZON * rng.lognormal(0, 0.15, 12)
            fec = 0.5 * n_ano * SAZON * rng.lognormal(0, 0.15, 12)
            pg = 400.0 * numcon / 1000 * max(n_ano - 0.5, 0) ** 2 * SAZON * rng.lognormal(0, 0.3, 12)
            if ano == ANO_ATUAL:
                verdade.append({"conjunto_id": cid, "viol_dec_real": float(dec.sum() > LIM_DEC),
                                "comp_resto_real": float(pg[MESES_PUBLICADOS:].sum())})
            for m in range(1, 13 if ano < ANO_ATUAL else MESES_PUBLICADOS + 1):
                base = {"SigAgente": f" {agente} ", "IdeConjUndConsumidoras": cid,
                        "DscConjUndConsumidoras": f"CONJ {k}", "AnoIndice": ano, "NumPeriodoIndice": m}
                ind += [{**base, "SigIndicador": "DEC", "VlrIndiceEnviado": dec[m - 1]},
                        {**base, "SigIndicador": "FEC", "VlrIndiceEnviado": fec[m - 1]},
                        {**base, "SigIndicador": "NumCon", "VlrIndiceEnviado": numcon},
                        {**base, "SigIndicador": "DECIP", "VlrIndiceEnviado": 999.0}]   # outro indicador: ignorar
                cb = {"IdeConjUndConsumidoras": cid, "AnoIndice": ano, "NumPeriodoIndice": m}
                comp += [{**cb, "SigIndicador": "PGDIC", "VlrIndiceEnviado": pg[m - 1]},
                         {**cb, "SigIndicador": "DIC", "VlrIndiceEnviado": 1e9}]        # não é PG*: ignorar
            if k != 0:                               # o conjunto 1000 fica sem limite publicado
                lim += [(cid, ano, "DEC", LIM_DEC), (cid, ano, "FEC", LIM_FEC)]
    # registro fora do intervalo mensal (por exemplo, apuração anual): tem que ficar de fora
    ind.append({"SigAgente": "AAA", "IdeConjUndConsumidoras": 1001, "DscConjUndConsumidoras": "CONJ 1",
                "AnoIndice": 2020, "NumPeriodoIndice": 13, "SigIndicador": "DEC", "VlrIndiceEnviado": 500.0})
    pd.DataFrame(ind).to_parquet(raw / build.IND[-1], index=False)
    pd.DataFrame(comp).to_parquet(raw / build.COMP[-1], index=False)
    with open(raw / build.LIMITE, "w", encoding="utf-8") as f:   # separador ";" e decimal ","
        f.write("IdeConjUndConsumidoras;AnoLimiteQualidade;SigIndicador;VlrLimite\n")
        f.writelines(f"{c};{a};{s};{str(v).replace('.', ',')}\n" for c, a, s, v in lim)
    return pd.DataFrame(verdade)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    verdade = _gerar(raw)
    mp = pytest.MonkeyPatch()

    def sem_rede(*a, **k):
        raise AssertionError("o teste de fumaça não pode usar a rede")

    mp.setattr(requests.sessions.Session, "request", sem_rede)
    try:
        base = build.base_mensal(raw)
        painel = F.painel(base)
        prev = M.prever(painel, ANO_ATUAL)
        yield {"raw": raw, "verdade": verdade, "base": base, "painel": painel, "prev": prev}
    finally:
        mp.undo()


def test_carrega_so_o_que_importa_dos_arquivos_da_aneel(pipeline):
    b = pipeline["base"]
    assert len(b) == N_CONJ * ((len(ANOS) - 1) * 12 + MESES_PUBLICADOS)
    assert b.ano.max() == ANO_ATUAL and b[b.ano == ANO_ATUAL].mes.max() == MESES_PUBLICADOS
    assert b.dec.max() < 10 and b.compensacao_rs.max() < 1e6     # sem o DECIP, o mês 13 e o indicador não-PG
    assert set(b.agente) == {"AAA", "BBB"}                       # espaços da sigla removidos
    sem_lim = b[b.conjunto_id == 1000]
    assert sem_lim.lim_dec.isna().all()
    com_lim = b[b.conjunto_id == 1001]
    assert (com_lim.lim_dec == LIM_DEC).all() and (com_lim.lim_fec == LIM_FEC).all()   # decimal com vírgula
    assert b.groupby(["conjunto_id", "ano", "mes"]).size().eq(1).all()


def test_painel_tem_alvo_nos_anos_fechados_e_nao_no_corrente(pipeline):
    p = pipeline["painel"]
    corrente = p[p.ano == ANO_ATUAL]
    assert set(corrente.mes_corte) == set(range(1, MESES_PUBLICADOS + 1))
    assert corrente.viol_dec.isna().all() and corrente.comp_resto_rs.isna().all()   # o futuro não existe
    fechados = p[(p.ano >= 2019) & (p.ano < ANO_ATUAL) & (p.conjunto_id != 1000)]
    assert fechados.viol_dec.notna().all() and set(fechados.viol_dec) == {0.0, 1.0}


def test_probabilidades_e_compensacao_sao_validas(pipeline):
    prev = pipeline["prev"]
    assert len(prev) == N_CONJ and prev.conjunto_id.is_unique
    assert (prev.mes_corte == MESES_PUBLICADOS).all()
    cols = ["p_viol_dec", "p_viol_fec", "p_viol_qualquer", "pred_comp_resto_rs"]
    assert prev[cols].notna().all().all()
    for c in cols[:3]:
        assert prev[c].between(0, 1).all()
    assert prev.pred_comp_resto_rs.ge(0).all()
    assert (prev.p_viol_qualquer >= prev[["p_viol_dec", "p_viol_fec"]].max(axis=1) - 1e-12).all()
    assert (prev.p_viol_qualquer <= prev.p_viol_dec + prev.p_viol_fec + 1e-12).all()


def test_previsao_enxerga_o_que_o_dado_sintetico_ensina(pipeline):
    """Não é um teste de qualidade do modelo: só garante que o caminho inteiro aprendeu algo."""
    d = pipeline["prev"].merge(pipeline["verdade"], on="conjunto_id")
    viol = d.viol_dec_real == 1
    assert 0.2 < viol.mean() < 0.8                                         # os dois casos existem
    assert d.p_viol_dec[viol].mean() > d.p_viol_dec[~viol].mean() + 0.3    # quem viola recebe risco maior
    assert d.p_viol_dec.sum() == pytest.approx(viol.sum(), rel=0.35)       # violações esperadas ~ reais
    ordem = d.p_viol_dec.rank()
    auc = (ordem[viol].sum() - viol.sum() * (viol.sum() + 1) / 2) / (viol.sum() * (~viol).sum())
    assert auc > 0.9
    total_real, total_prev = d.comp_resto_real.sum(), d.pred_comp_resto_rs.sum()
    assert total_real / 3 < total_prev < total_real * 3                    # mesma ordem de grandeza
    assert d[["pred_comp_resto_rs", "comp_resto_real"]].corr(method="spearman").iloc[0, 1] > 0.5


def test_lista_e_ranking_fecham_com_a_previsao(pipeline):
    prev = pipeline["prev"]
    lista = R.lista_conjuntos(prev)
    assert len(lista) == N_CONJ and lista.p_viol_qualquer.is_monotonic_decreasing
    rank = R.ranking_distribuidoras(prev)
    assert set(rank.agente) == {"AAA", "BBB"} and rank.conjuntos.sum() == N_CONJ
    assert rank.violacoes_dec_esperadas.sum() == pytest.approx(prev.p_viol_dec.sum())
    assert rank.compensacao_esperada_resto_rs.sum() == pytest.approx(prev.pred_comp_resto_rs.sum())


def test_resultado_e_deterministico(pipeline):
    de_novo = M.prever(F.painel(build.base_mensal(pipeline["raw"])), ANO_ATUAL)
    pd.testing.assert_frame_equal(de_novo, pipeline["prev"])
