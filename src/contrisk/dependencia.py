"""Probabilidade de violar DEC *ou* FEC, com a dependência entre os dois estimada nos dados.

Os dois classificadores dão probabilidades marginais, P(DEC) e P(FEC). Para P(DEC ou FEC) falta saber quanto
os dois eventos andam juntos, depois de descontado o que os próprios modelos já explicam. Em vez de supor
independência, aqui a dependência vira um único número, ``rho``, de uma cópula gaussiana:

    P(DEC e FEC) = Φ2(Φ⁻¹(P_DEC), Φ⁻¹(P_FEC); rho)        P(DEC ou FEC) = P_DEC + P_FEC − P(DEC e FEC)

Com ``rho = 0`` volta a fórmula antiga, 1 − (1 − P_DEC)(1 − P_FEC). ``rho`` é estimado por máxima
verossimilhança das quatro combinações (viola/não viola, DEC/FEC) nas previsões **fora da amostra** do
backtest de anos anteriores ao ano previsto (``model.backtest_dependencia``).

A soma das probabilidades marginais (violações esperadas de DEC ou de FEC) não depende de ``rho``: é o
valor esperado da contagem, se as marginais forem calibradas. A dependência muda P(DEC ou FEC) e, com ela,
o número de conjuntos de risco alto.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri

log = logging.getLogger(__name__)

EPS = 1e-9
MIN_LINHAS = 200
LIMITES_RHO = (-0.95, 0.95)
_X, _W = np.polynomial.legendre.leggauss(96)      # quadratura de Gauss-Legendre
_LIM_INF = -9.0                                    # φ(-9) ≈ 1e-19: cauda desprezível


def _marginais(p_dec, p_fec) -> tuple[np.ndarray, np.ndarray]:
    return (np.clip(np.asarray(p_dec, dtype=float), EPS, 1 - EPS),
            np.clip(np.asarray(p_fec, dtype=float), EPS, 1 - EPS))


def p_ambos(p_dec, p_fec, rho: float) -> np.ndarray:
    """P(violar DEC e FEC) sob cópula gaussiana com correlação ``rho`` e as marginais dadas.

    Φ2(h, k; ρ) = ∫ φ(x) Φ((k − ρx)/√(1−ρ²)) dx de −∞ até h, com h = Φ⁻¹(P_DEC) e k = Φ⁻¹(P_FEC) (violar
    é a cauda de cima, mas a correlação é a mesma se os dois eixos forem invertidos).
    """
    pd_, pf = _marginais(p_dec, p_fec)
    if rho == 0:
        return pd_ * pf
    if not -1 < rho < 1:
        raise ValueError("rho precisa estar entre -1 e 1 (exclusive)")
    h, k = ndtri(pd_), ndtri(pf)
    meio, meia = (h + _LIM_INF) / 2, (h - _LIM_INF) / 2
    x = meio[..., None] + meia[..., None] * _X
    integrando = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi) * ndtr((k[..., None] - rho * x) / np.sqrt(1 - rho * rho))
    ambos = meia * (integrando * _W).sum(axis=-1)
    return np.clip(ambos, np.maximum(0.0, pd_ + pf - 1), np.minimum(pd_, pf))   # limites de Fréchet


def p_qualquer(p_dec, p_fec, rho: float = 0.0) -> np.ndarray:
    """P(violar DEC ou FEC). Com ``rho = 0`` é a fórmula da independência, 1 − (1 − P_DEC)(1 − P_FEC)."""
    p_dec, p_fec = np.asarray(p_dec, dtype=float), np.asarray(p_fec, dtype=float)
    if rho == 0:
        return 1 - (1 - p_dec) * (1 - p_fec)
    pd_, pf = _marginais(p_dec, p_fec)
    return np.clip(pd_ + pf - p_ambos(pd_, pf, rho), np.maximum(pd_, pf), np.minimum(1.0, pd_ + pf))


def log_verossimilhanca(y_dec, y_fec, p_dec, p_fec, rho: float) -> np.ndarray:
    """log P(y_dec, y_fec) por linha, nas quatro combinações possíveis."""
    y_dec, y_fec = np.asarray(y_dec) == 1, np.asarray(y_fec) == 1
    pd_, pf = _marginais(p_dec, p_fec)
    ambos = p_ambos(pd_, pf, rho)
    celula = np.where(y_dec & y_fec, ambos,
                      np.where(y_dec, pd_ - ambos, np.where(y_fec, pf - ambos, 1 - pd_ - pf + ambos)))
    return np.log(np.clip(celula, 1e-12, None))


def estimar_rho(y_dec, y_fec, p_dec, p_fec) -> float:
    """``rho`` de máxima verossimilhança. Ignora linhas com alvo ou probabilidade ausente.

    Com poucas linhas (< ``MIN_LINHAS``) devolve 0 (independência) e avisa.
    """
    y_dec, y_fec, p_dec, p_fec = (np.asarray(a, dtype=float) for a in (y_dec, y_fec, p_dec, p_fec))
    ok = ~(np.isnan(y_dec) | np.isnan(y_fec) | np.isnan(p_dec) | np.isnan(p_fec))
    if ok.sum() < MIN_LINHAS:
        log.warning("só %d linhas para estimar a dependência DEC x FEC; usando independência", ok.sum())
        return 0.0
    y1, y2, a, b = y_dec[ok], y_fec[ok], p_dec[ok], p_fec[ok]
    r = minimize_scalar(lambda rho: -log_verossimilhanca(y1, y2, a, b, rho).sum(), bounds=LIMITES_RHO,
                        method="bounded", options={"xatol": 1e-4})
    return float(r.x)


def salvar(info: dict, path: Path) -> None:
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_rho(path: Path) -> float | None:
    """``rho`` gravado pelo backtest, se o critério de adoção foi atendido; senão ``None`` (independência).

    Nunca levanta exceção: arquivo ausente, ilegível ou com valor inválido cai na independência, com aviso.
    """
    try:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        rho = float(info["rho"])
        adotar = bool(info["adotar"])
    except FileNotFoundError:
        log.warning("%s não existe; P(DEC ou FEC) com independência", path)
        return None
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("%s ilegível; P(DEC ou FEC) com independência", path, exc_info=True)
        return None
    if not adotar:
        log.info("a dependência estimada não passou no critério de adoção; P(DEC ou FEC) com independência")
        return None
    if not np.isfinite(rho) or not -1 < rho < 1:
        log.warning("rho inválido em %s (%r); P(DEC ou FEC) com independência", path, rho)
        return None
    return rho
