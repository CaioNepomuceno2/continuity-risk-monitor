"""Leitura dos arquivos da ANEEL: registros duplicados não podem inflar as compensações."""
import pandas as pd
import pytest

from contrisk import build

COLS = ["DatGeracaoConjuntoDados", "SigAgente", "NumCNPJ", "IdeConjUndConsumidoras", "DscConjUndConsumidoras",
        "SigIndicador", "AnoIndice", "NumPeriodoIndice", "VlrIndiceEnviado"]


def _reg(conj, ind, valor, ano=2024, mes=3, cnpj=111, agente="AAA"):
    return ["2026-11-05", agente, cnpj, conj, f"C{conj}", ind, ano, mes, valor]


def _linhas():
    return [
        _reg(1, "PGUCAT", 100.0), _reg(1, "PGUCBTU", 50.0),                  # conjunto 1: 150 no mês
        _reg(2, "PGUCAT", 72.05), _reg(2, "PGUCAT", 0.0),                    # mesma chave, valores diferentes: os dois contam
        _reg(3, "PGUCAT", 10.0, cnpj=111), _reg(3, "PGUCAT", 10.0, cnpj=222),  # outra empresa: conta
        _reg(4, "QTUCAT", 999.0), _reg(4, "PGUCAT", 5.0, mes=13),            # não é PG*; mês fora de 1-12
    ]


def _compensacao(tmp_path, linhas):
    pd.DataFrame(linhas, columns=COLS).to_parquet(tmp_path / build.COMP[-1], index=False)
    return build.compensacao_mensal(tmp_path).set_index("conjunto_id").compensacao_rs


def test_compensacao_soma_os_registros_pg_do_mes(tmp_path):
    c = _compensacao(tmp_path, _linhas())
    assert c.to_dict() == pytest.approx({1: 150.0, 2: 72.05, 3: 20.0})


def test_arquivo_com_todos_os_registros_duplicados_da_o_mesmo_resultado(tmp_path):
    limpo = _compensacao(tmp_path, _linhas())
    duplicado = _compensacao(tmp_path, _linhas() * 2)          # como nos arquivos da ANEEL de 05/11/2026
    quadruplicado = _compensacao(tmp_path, _linhas() * 4)
    assert duplicado.to_dict() == pytest.approx(limpo.to_dict())
    assert quadruplicado.to_dict() == pytest.approx(limpo.to_dict())


def test_dec_fec_e_numcon_ignoram_repeticoes(tmp_path):
    ind = [_reg(1, "DEC", 2.5), _reg(1, "FEC", 1.0), _reg(1, "NumCon", 1000.0)]
    pd.DataFrame(ind * 2, columns=COLS).to_parquet(tmp_path / build.IND[-1], index=False)
    m = build.mensal(tmp_path).iloc[0]
    assert (m.dec, m.fec, m.numcon) == (2.5, 1.0, 1000.0)
