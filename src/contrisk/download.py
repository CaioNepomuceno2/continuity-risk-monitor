"""Download dos indicadores coletivos de continuidade da ANEEL (dados abertos, CKAN)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

CKAN = "https://dadosabertos.aneel.gov.br/api/3/action/package_show"
DATASET = "indicadores-coletivos-de-continuidade-dec-e-fec"
ARQUIVOS = [
    "indicadores-continuidade-coletivos-2010-2019.parquet",
    "indicadores-continuidade-coletivos-2020-2029.parquet",
    "indicadores-continuidade-coletivos-compensacao-2010-2019.parquet",
    "indicadores-continuidade-coletivos-compensacao-2020-2029.parquet",
    "indicadores-continuidade-coletivos-limite.csv",
    "dominio-indicadores.csv",
]
# arquivos que mudam todo mês (o robô sempre baixa de novo)
ATUALIZADOS = {"indicadores-continuidade-coletivos-2020-2029.parquet",
               "indicadores-continuidade-coletivos-compensacao-2020-2029.parquet",
               "indicadores-continuidade-coletivos-limite.csv"}

log = logging.getLogger(__name__)


def resource_urls() -> dict[str, str]:
    r = requests.get(CKAN, params={"id": DATASET}, timeout=60)
    r.raise_for_status()
    return {res["url"].rsplit("/", 1)[-1]: res["url"] for res in r.json()["result"]["resources"]}


def _baixar(url: str, out: Path, retries: int = 4) -> None:
    for k in range(retries):
        try:
            with requests.get(url, stream=True, timeout=1200) as r:
                r.raise_for_status()
                tmp = out.with_suffix(out.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.writelines(r.iter_content(1 << 20))
                tmp.replace(out)
                return
        except requests.RequestException as e:
            log.warning("falha ao baixar %s (%s), tentativa %d", url, e, k + 1)
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"não consegui baixar {url}")


def download(dest: Path, atualizar: bool = False) -> list[Path]:
    """Baixa os arquivos que faltam; com ``atualizar``, baixa de novo os que mudam todo mês."""
    dest.mkdir(parents=True, exist_ok=True)
    urls = resource_urls()
    paths = []
    for nome in ARQUIVOS:
        if nome not in urls:
            raise RuntimeError(f"{nome} não está mais no dataset {DATASET}")
        out = dest / nome
        if not out.exists() or (atualizar and nome in ATUALIZADOS):
            log.info("baixando %s", nome)
            _baixar(urls[nome], out)
        paths.append(out)
    return paths
