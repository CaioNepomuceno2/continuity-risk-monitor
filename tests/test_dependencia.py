"""Testes da probabilidade conjunta de violar DEC e FEC (cópula gaussiana calibrada no backtest)."""
import json
import logging

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri
from scipy.stats import multivariate_normal

from contrisk import dependencia as D
from contrisk import features as F
from contrisk import model as M


# --------------------------------------------------------------------------- matemática
@pytest.mark.parametrize("pd_,pf,rho", [(0.3, 0.6, 0.5), (0.05, 0.9, -0.4), (0.5, 0.5, 0.9), (0.97, 0.02, 0.3),
                                        (0.001, 0.2, 0.7)])
def test_p_ambos_bate_com_a_normal_bivariada_de_referencia(pd_, pf, rho):
    ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([ndtri(pd_), ndtri(pf)])
    assert float(D.p_ambos(pd_, pf, rho)) == pytest.approx(ref, abs=2e-6)


def test_p_ambos_respeita_independencia_e_limites_de_frechet():
    pd_, pf = np.array([0.1, 0.4, 0.8, 0.95]), np.array([0.2, 0.7, 0.3, 0.9])
    np.testing.assert_allclose(D.p_ambos(pd_, pf, 0.0), pd_ * pf)
    for rho in (-0.95, -0.5, 0.5, 0.95):
        a = D.p_ambos(pd_, pf, rho)
        assert (a >= np.maximum(0, pd_ + pf - 1) - 1e-12).all() and (a <= np.minimum(pd_, pf) + 1e-12).all()
    # mais correlação positiva, mais violações juntas
    assert (D.p_ambos(pd_, pf, 0.8) > D.p_ambos(pd_, pf, 0.2)).all()
    assert (D.p_ambos(pd_, pf, 0.2) > D.p_ambos(pd_, pf, 0.0)).all()
    assert (D.p_ambos(pd_, pf, -0.5) < D.p_ambos(pd_, pf, 0.0)).all()
    # quase perfeita: tende ao mínimo
    assert D.p_ambos(0.3, 0.6, 0.95) == pytest.approx(0.3, abs=0.03)


def test_p_ambos_aceita_matrizes_e_probabilidades_nos_extremos():
    p = np.array([[0.0, 1.0], [0.5, 1e-12]])
    a = D.p_ambos(p, p[::-1], 0.5)
    assert a.shape == (2, 2) and np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    with pytest.raises(ValueError):
        D.p_ambos(0.5, 0.5, 1.0)


def test_p_qualquer_com_rho_zero_e_a_formula_antiga_exata():
    pd_, pf = np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.7, 0.3])
    np.testing.assert_array_equal(D.p_qualquer(pd_, pf), 1 - (1 - pd_) * (1 - pf))
    np.testing.assert_array_equal(D.p_qualquer(pd_, pf, 0.0), 1 - (1 - pd_) * (1 - pf))


def test_p_qualquer_com_correlacao_positiva_fica_abaixo_da_independencia():
    pd_, pf = np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.7, 0.3])
    q = D.p_qualquer(pd_, pf, 0.6)
    assert (q < D.p_qualquer(pd_, pf)).all()
    assert (q >= np.maximum(pd_, pf) - 1e-12).all() and (q <= np.minimum(1, pd_ + pf) + 1e-12).all()
    assert (D.p_qualquer(pd_, pf, -0.6) > D.p_qualquer(pd_, pf)).all()


# --------------------------------------------------------------------------- estimação
def _simular(n, rho, seed=0):
    """Violações DEC/FEC de uma cópula gaussiana com marginais heterogêneas conhecidas."""
    rng = np.random.default_rng(seed)
    pd_, pf = rng.beta(0.6, 0.9, n), rng.beta(0.5, 1.2, n)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    vd = (z[:, 0] > ndtri(1 - pd_)).astype(float)
    vf = (z[:, 1] > ndtri(1 - pf)).astype(float)
    return vd, vf, pd_, pf


@pytest.mark.parametrize("rho", [-0.4, 0.0, 0.6])
def test_estimar_rho_recupera_a_dependencia_verdadeira(rho):
    vd, vf, pd_, pf = _simular(20000, rho)
    assert D.estimar_rho(vd, vf, pd_, pf) == pytest.approx(rho, abs=0.06)


def test_estimar_rho_ignora_ausentes_e_cai_em_zero_com_poucos_dados(caplog):
    vd, vf, pd_, pf = _simular(5000, 0.6)
    vd[:300] = np.nan
    pf[300:500] = np.nan
    assert D.estimar_rho(vd, vf, pd_, pf) == pytest.approx(0.6, abs=0.08)
    with caplog.at_level(logging.WARNING):
        assert D.estimar_rho(vd[:50], vf[:50], pd_[:50], pf[:50]) == 0.0
    assert "independência" in caplog.text


# --------------------------------------------------------------------------- arquivo de adoção
def test_carregar_rho_so_devolve_valor_quando_adotado_e_valido(tmp_path, caplog):
    f = tmp_path / "dep.json"
    assert D.carregar_rho(f) is None                                  # não existe
    f.write_text("{não é json", encoding="utf-8")
    assert D.carregar_rho(f) is None                                  # corrompido
    D.salvar({"rho": 0.5}, f)
    assert D.carregar_rho(f) is None                                  # sem a decisão de adoção
    D.salvar({"rho": 0.5, "adotar": False}, f)
    assert D.carregar_rho(f) is None                                  # não passou no critério
    D.salvar({"rho": 1.5, "adotar": True}, f)
    assert D.carregar_rho(f) is None                                  # valor impossível
    D.salvar({"rho": "abc", "adotar": True}, f)
    assert D.carregar_rho(f) is None
    D.salvar({"rho": 0.5, "adotar": True}, f)
    assert D.carregar_rho(f) == 0.5
    assert json.loads(f.read_text(encoding="utf-8"))["adotar"] is True


# --------------------------------------------------------------------------- backtest sem vazamento
def _prev_sintetico(rho_verdadeiro, anos=range(2016, 2026), n=300, seed=1):
    """Previsões 'fora da amostra' de um backtest: marginais conhecidas, alvos de uma cópula com rho dado."""
    quadros = []
    for k, ano in enumerate(anos):
        for m in (3, 6, 9):
            vd, vf, pd_, pf = _simular(n, rho_verdadeiro, seed=seed + 100 * k + m)
            quadros.append(pd.DataFrame({"ano": ano, "mes_corte": m, "conjunto_id": np.arange(n),
                                         "viol_dec": vd, "viol_fec": vf, "p_viol_dec": pd_, "p_viol_fec": pf}))
    return pd.concat(quadros, ignore_index=True)


def test_rho_de_um_ano_de_teste_nao_enxerga_o_proprio_ano_nem_os_seguintes():
    prev = _prev_sintetico(0.6)
    anos_teste = [2021, 2022, 2023, 2024, 2025]
    _, _, _, info = M.backtest_dependencia(prev, anos_teste, 2016)
    # embaralha os alvos de 2023 em diante: o rho dos anos até 2023 não pode mudar
    sujo = prev.copy()
    sujo.loc[sujo.ano >= 2023, ["viol_dec", "viol_fec"]] = 1.0 - sujo.loc[sujo.ano >= 2023,
                                                                         ["viol_dec", "viol_fec"]].values
    _, _, _, info2 = M.backtest_dependencia(sujo, anos_teste, 2016)
    for t in (2021, 2022, 2023):
        assert info2["rho_por_ano_de_teste"][t] == info["rho_por_ano_de_teste"][t]
    assert info2["rho_por_ano_de_teste"][2024] != info["rho_por_ano_de_teste"][2024]   # 2023 já entra
    # a calibração dos anos anteriores a 2021 usa só 2016-2020
    assert info["anos_calibracao"] == [2016, 2025]


def test_backtest_dependencia_adota_a_copula_quando_ha_dependencia_real():
    prev = _prev_sintetico(0.7)
    corte, ano, faixas, info = M.backtest_dependencia(prev, [2021, 2022, 2023, 2024, 2025], 2016)
    assert info["rho"] == pytest.approx(0.7, abs=0.06) and info["adotar"] is True
    ag = ano[ano.ano == "todos"].set_index("metodo")
    assert ag.loc["copula", "brier"] < ag.loc["independencia", "brier"]
    assert ag.loc["copula", "loglik_conjunta"] > ag.loc["independencia", "loglik_conjunta"]
    # a soma das marginais não depende de rho: a contagem esperada de DEC e FEC é igual nos dois métodos
    assert ag.loc["copula", "esperado_dec"] == pytest.approx(ag.loc["independencia", "esperado_dec"])
    # a contagem esperada de "DEC ou FEC" chega mais perto do observado com a cópula
    obs = ag.loc["copula", "observado_qualquer"]
    assert abs(ag.loc["copula", "esperado_qualquer"] - obs) < abs(ag.loc["independencia", "esperado_qualquer"] - obs)
    assert set(corte.metodo) == {"independencia", "copula"} and set(corte.mes_corte) == {3, 6, 9}
    assert faixas.groupby("metodo").n.sum().eq(len(prev[prev.ano >= 2021])).all()


def test_backtest_dependencia_nao_adota_sem_dependencia():
    prev = _prev_sintetico(0.0)
    _, _, _, info = M.backtest_dependencia(prev, [2021, 2022, 2023, 2024, 2025], 2016)
    assert abs(info["rho"]) < 0.1 and info["adotar"] is False


# --------------------------------------------------------------------------- integração com prever()
class _Const:
    """Modelo de mentira: probabilidade constante, para isolar a combinação de DEC e FEC."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])

    def predict(self, X):
        return np.full(len(X), 10.0)


def _painel_minimo():
    linhas = [{"agente": "XYZ", "conjunto_id": c, "conjunto": f"C{c}", "ano": a, "mes": m, "dec": 1.0 * c,
               "fec": 0.5 * c, "numcon": 1000.0, "compensacao_rs": 100.0, "lim_dec": 10.0, "lim_fec": 8.0}
              for c in (1, 2) for a in (2019, 2020, 2021) for m in range(1, 13)]
    return F.painel(pd.DataFrame(linhas))


def test_prever_usa_independencia_sem_rho_e_a_copula_com_rho(monkeypatch):
    monkeypatch.setattr(M, "fit_classificador", lambda tr, alvo, seed=0: _Const(0.6 if alvo == "viol_dec" else 0.5))
    monkeypatch.setattr(M, "fit_compensacao", lambda tr, seed=0: _Const(0))
    painel = _painel_minimo()
    indep = M.prever(painel, 2021, seeds=1)
    assert indep.p_viol_qualquer.tolist() == pytest.approx([1 - 0.4 * 0.5] * 2)
    assert M.prever(painel, 2021, seeds=1, rho=0.0).p_viol_qualquer.tolist() == indep.p_viol_qualquer.tolist()
    com = M.prever(painel, 2021, seeds=1, rho=0.6)
    esperado = float(D.p_qualquer(0.6, 0.5, 0.6))
    assert com.p_viol_qualquer.tolist() == pytest.approx([esperado] * 2) and esperado < 0.8
    assert com.p_viol_dec.tolist() == indep.p_viol_dec.tolist()      # as marginais não mudam
