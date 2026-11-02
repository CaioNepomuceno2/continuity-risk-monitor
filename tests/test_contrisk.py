"""Testes com dados sintéticos: anualização, cortes temporais, métricas e calibração."""
import numpy as np
import pandas as pd
import pytest

from contrisk import features as F
from contrisk import model as M


def _base(anos=(2018, 2019, 2020, 2021), conjuntos=(1, 2), dec_mes=1.0, lim=10.0, meses_ultimo=12):
    linhas = []
    for c in conjuntos:
        for a in anos:
            n = meses_ultimo if a == max(anos) else 12
            for m in range(1, n + 1):
                linhas.append({"agente": "XYZ", "conjunto_id": c, "conjunto": f"C{c}", "ano": a, "mes": m,
                               "dec": dec_mes * c, "fec": 0.5 * c, "numcon": 1000.0,
                               "compensacao_rs": 100.0 * m, "lim_dec": lim, "lim_fec": 8.0})
    return pd.DataFrame(linhas)


def test_anual_soma_12_meses_e_compara_com_limite():
    a = F.anual(_base())
    c1 = a[(a.conjunto_id == 1) & (a.ano == 2019)].iloc[0]
    c2 = a[(a.conjunto_id == 2) & (a.ano == 2019)].iloc[0]
    assert c1.dec_anual == 12.0 and c1.viol_dec == 1.0          # 12 > 10
    assert c2.fec_anual == 12.0 and c2.viol_fec == 1.0          # 12 > 8
    assert F.anual(_base(lim=12.0)).query("conjunto_id == 1").viol_dec.eq(0).all()  # igual ao limite não viola


def test_ano_incompleto_nao_entra_no_fechamento():
    a = F.anual(_base(meses_ultimo=7))
    assert 2021 not in set(a.ano)


def test_limite_ausente_vira_nan():
    b = _base()
    b.loc[b.ano == 2019, "lim_dec"] = np.nan
    a = F.anual(b)
    assert a.query("ano == 2019").viol_dec.isna().all()


def test_painel_usa_so_meses_ate_o_corte_e_anos_anteriores():
    b = _base()
    # um evento enorme em novembro de 2020 não pode aparecer nas features de junho de 2020
    b.loc[(b.ano == 2020) & (b.mes == 11), "dec"] = 500.0
    p = F.painel(b, cortes=[6])
    r = p[(p.conjunto_id == 1) & (p.ano == 2020)].iloc[0]
    assert r.dec_acum == 6.0
    assert r.dec_ult3 == 3.0
    assert r.dec_anual_ant1 == 12.0                              # fechamento de 2019
    assert r.hist_frac_dec == pytest.approx(0.5)                 # perfil de anos < 2020
    assert r.dec_anual == 511.0                                  # o alvo enxerga o ano inteiro
    # e o evento de 2020 só entra no histórico de 2021
    r21 = p[(p.conjunto_id == 1) & (p.ano == 2021)].iloc[0]
    assert r21.dec_anual_ant1 == 511.0


def test_alvo_de_compensacao_e_o_resto_do_ano():
    p = F.painel(_base(), cortes=[9])
    r = p[(p.conjunto_id == 1) & (p.ano == 2020)].iloc[0]
    assert r.compensacao_rs_acum == 100.0 * sum(range(1, 10))
    assert r.comp_resto_rs == 100.0 * (10 + 11 + 12)


def test_feature_cols_sem_alvos_nem_previsoes():
    p = F.painel(_base(), cortes=[6])
    p["p_viol_dec"] = 0.5
    p["pred_comp_resto_rs"] = 1.0
    cols = F.feature_cols(p)
    assert not set(cols) & set(F.ALVOS)
    assert "p_viol_dec" not in cols and "pred_comp_resto_rs" not in cols


def test_metricas_e_calibracao():
    y = np.array([1, 1, 0, 0, 1, 0], dtype=float)
    perfeito = y.copy()
    m = M.metricas(y, perfeito, perfeito > 0.5, perfeito, np.ones(6))
    assert m["auc"] == 1.0 and m["precisao"] == 1.0 and m["recall"] == 1.0 and m["brier"] == 0.0
    assert M.ece(y, np.full(6, 0.5)) == pytest.approx(0.0)       # 50% previsto, 50% observado
    assert M.ece(y, np.full(6, 0.9)) == pytest.approx(0.4)


def test_metricas_ignoram_escore_ausente():
    y = np.array([1, 0, 1], dtype=float)
    esc = np.array([0.9, np.nan, 0.1])
    m = M.metricas(y, esc, esc > 0.5, None, np.ones(3))
    assert m["n"] == 2 and m["sinalizados"] == 1
