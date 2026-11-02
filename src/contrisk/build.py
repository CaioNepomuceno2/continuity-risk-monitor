"""Bases mensais (DEC, FEC, consumidores, compensações) e limites anuais por conjunto.

Regras verificadas nos próprios dados (ver docs/dados.md):

* ``DEC`` e ``FEC`` publicados são os indicadores apurados para comparação com o limite: batem com
  ``DECIP + DECIND`` (interrupções de origem interna, programadas e não programadas não expurgáveis).
* O indicador anual é a soma dos 12 valores mensais.
* Compensações: soma dos indicadores ``PG*`` (R$ pagos por violação de DIC/FIC/DMIC/DICRI/DISE,
  mensal, trimestral e anual) no mês em que foram informados.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

IND = ["indicadores-continuidade-coletivos-2010-2019.parquet",
       "indicadores-continuidade-coletivos-2020-2029.parquet"]
COMP = ["indicadores-continuidade-coletivos-compensacao-2010-2019.parquet",
        "indicadores-continuidade-coletivos-compensacao-2020-2029.parquet"]
LIMITE = "indicadores-continuidade-coletivos-limite.csv"


def _lista(raw: Path, nomes: list[str]) -> str:
    return "[" + ", ".join(f"'{raw / n}'" for n in nomes if (raw / n).exists()) + "]"


def mensal(raw: Path, ano_min: int = 2011) -> pd.DataFrame:
    """Uma linha por conjunto x ano x mês com DEC, FEC e número de consumidores."""
    return duckdb.sql(f"""
        select trim(any_value(SigAgente)) as agente, IdeConjUndConsumidoras as conjunto_id,
               any_value(DscConjUndConsumidoras) as conjunto,
               AnoIndice as ano, NumPeriodoIndice as mes,
               max(case when SigIndicador = 'DEC' then VlrIndiceEnviado end) as dec,
               max(case when SigIndicador = 'FEC' then VlrIndiceEnviado end) as fec,
               max(case when SigIndicador = 'NumCon' then VlrIndiceEnviado end) as numcon
        from read_parquet({_lista(raw, IND)}, union_by_name = true)
        where SigIndicador in ('DEC', 'FEC', 'NumCon') and AnoIndice >= {ano_min}
          and NumPeriodoIndice between 1 and 12
        group by conjunto_id, ano, mes
    """).df()


def compensacao_mensal(raw: Path, ano_min: int = 2011) -> pd.DataFrame:
    """R$ pagos em compensações por conjunto x ano x mês."""
    return duckdb.sql(f"""
        select IdeConjUndConsumidoras as conjunto_id, AnoIndice as ano, NumPeriodoIndice as mes,
               sum(VlrIndiceEnviado) as compensacao_rs
        from read_parquet({_lista(raw, COMP)}, union_by_name = true)
        where SigIndicador like 'PG%' and AnoIndice >= {ano_min} and NumPeriodoIndice between 1 and 12
        group by all
    """).df()


def limites(raw: Path) -> pd.DataFrame:
    """Limites anuais de DEC e FEC por conjunto (um valor por conjunto x ano)."""
    return duckdb.sql(f"""
        select IdeConjUndConsumidoras as conjunto_id, AnoLimiteQualidade as ano,
               max(case when SigIndicador = 'DEC' then VlrLimite end) as lim_dec,
               max(case when SigIndicador = 'FEC' then VlrLimite end) as lim_fec
        from read_csv('{raw / LIMITE}', delim = ';', decimal_separator = ',', header = true,
                      types = {{'VlrLimite': 'DOUBLE'}})
        group by all
    """).df()


def base_mensal(raw: Path, ano_min: int = 2011) -> pd.DataFrame:
    """Indicadores + compensações + limites do ano, por conjunto x ano x mês."""
    m = mensal(raw, ano_min)
    c = compensacao_mensal(raw, ano_min)
    lim = limites(raw)
    df = m.merge(c, on=["conjunto_id", "ano", "mes"], how="left").merge(lim, on=["conjunto_id", "ano"], how="left")
    df["compensacao_rs"] = df.compensacao_rs.fillna(0.0)
    return df.sort_values(["conjunto_id", "ano", "mes"], ignore_index=True)
