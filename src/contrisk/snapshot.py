"""Cópia datada dos arquivos brutos da ANEEL, para formar com o tempo a base "como foi publicada".

A ANEEL regenera os arquivos com frequência e meses passados podem ser revisados pelas distribuidoras. O
backtest usa a base atual, então não vê essas revisões. Guardar o que foi baixado a cada execução permite,
com o tempo, refazer o backtest com a base que existia na época.

Como funciona (ver ``docs/dados.md``):

1. ``preparar`` compara o sha256 de cada arquivo baixado com o último registro de ``snapshots/manifesto.csv``
   e copia para ``data/snapshots/<tag>/`` só o que mudou (CSV vai em gzip; Parquet, que já é comprimido,
   vai idêntico byte a byte).
2. ``publicar`` anexa essa pasta a uma *release* do GitHub (``gh release create``) e só então acrescenta
   as linhas ao manifesto, que é versionado no git (poucos KB por mês). Os arquivos pesados ficam nos
   *assets* da release, fora do histórico do repositório.

Nenhuma das duas etapas pode derrubar o robô mensal: ``preparar_sem_falhar`` engole e registra qualquer
erro, e o workflow roda ``publicar`` com ``continue-on-error``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import requests

from contrisk.download import ARQUIVOS, CKAN, DATASET

log = logging.getLogger(__name__)

COLUNAS = ["data_utc", "tag", "arquivo", "bytes_originais", "sha256", "ckan_last_modified", "asset", "bytes_asset"]
COMPRIMIR = {".csv"}      # Parquet já é comprimido: gzip quase não ganha e tira a identidade byte a byte
LOTE = "lote.csv"         # linhas do manifesto que a pasta preparada vai acrescentar após o upload


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def ler_manifesto(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ultimos_hashes(linhas: list[dict]) -> dict[str, str]:
    """sha256 do último registro de cada arquivo."""
    return {r["arquivo"]: r["sha256"] for r in linhas}


def metadados_ckan() -> dict[str, str]:
    """``last_modified`` de cada recurso segundo o CKAN da ANEEL. Melhor esforço: em caso de erro, vazio."""
    try:
        r = requests.get(CKAN, params={"id": DATASET}, timeout=60)
        r.raise_for_status()
        return {x["url"].rsplit("/", 1)[-1]: str(x.get("last_modified") or "") for x in r.json()["result"]["resources"]}
    except Exception as e:  # noqa: BLE001 - sem esses metadados o snapshot continua valendo
        log.warning("não consegui os metadados do CKAN (%s); o manifesto fica sem ckan_last_modified", e)
        return {}


def _gravar_csv(path: Path, linhas: list[dict], anexar: bool = False) -> None:
    novo = not (anexar and Path(path).exists())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if anexar else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS, lineterminator="\n")
        if novo:
            w.writeheader()
        w.writerows(linhas)


def preparar(raw: Path, saida: Path, manifesto: Path, agora: datetime | None = None,
             meta: dict[str, str] | None = None) -> Path | None:
    """Prepara em ``saida/<tag>/`` os arquivos de ``raw`` novos ou alterados desde o último registro.

    Devolve a pasta preparada, ou ``None`` se nenhum arquivo mudou. Não mexe no manifesto.
    """
    agora = agora or datetime.now(UTC)
    tag = "dados-aneel-" + agora.strftime("%Y%m%d-%H%M")
    meta = metadados_ckan() if meta is None else meta
    anteriores = ultimos_hashes(ler_manifesto(manifesto))
    pasta, linhas = Path(saida) / tag, []
    for nome in ARQUIVOS:
        origem = Path(raw) / nome
        if not origem.exists():
            continue
        h = sha256(origem)
        if anteriores.get(nome) == h:
            log.info("snapshot: %s não mudou desde o último registro", nome)
            continue
        pasta.mkdir(parents=True, exist_ok=True)
        if origem.suffix in COMPRIMIR:
            asset = nome + ".gz"
            with open(origem, "rb") as f, gzip.GzipFile(pasta / asset, "wb", mtime=0) as g:   # mtime=0: reprodutível
                shutil.copyfileobj(f, g, 1 << 20)
        else:
            asset = nome
            shutil.copyfile(origem, pasta / asset)
        linhas.append({"data_utc": agora.strftime("%Y-%m-%d %H:%M"), "tag": tag, "arquivo": nome,
                       "bytes_originais": origem.stat().st_size, "sha256": h,
                       "ckan_last_modified": meta.get(nome, ""), "asset": asset,
                       "bytes_asset": (pasta / asset).stat().st_size})
    if not linhas:
        return None
    _gravar_csv(pasta / LOTE, linhas)
    return pasta


def preparar_sem_falhar(raw: Path, saida: Path, manifesto: Path, **kw) -> Path | None:
    """``preparar`` que nunca levanta exceção (usado pelo robô mensal)."""
    try:
        pasta = preparar(raw, saida, manifesto, **kw)
    except Exception:
        log.exception("snapshot dos dados brutos falhou; o robô segue sem ele")
        return None
    if pasta is None:
        log.info("snapshot: nenhum arquivo bruto mudou desde o último registro")
    else:
        n = len(ler_manifesto(pasta / LOTE))
        log.info("snapshot: %d arquivo(s) novo(s) ou alterado(s) preparado(s) em %s", n, pasta)
    return pasta


def pasta_pendente(saida: Path) -> Path | None:
    """A pasta preparada mais recente que ainda tem ``lote.csv`` (ainda não publicada)."""
    pastas = sorted(p for p in Path(saida).glob("dados-aneel-*") if (p / LOTE).exists())
    return pastas[-1] if pastas else None


def comando_release(pasta: Path, linhas: list[dict], gh: str = "gh") -> list[str]:
    tag = linhas[0]["tag"]
    nota = (f"Cópia datada dos arquivos brutos do dataset `{DATASET}` da ANEEL, baixados em "
            f"{linhas[0]['data_utc']} UTC. Só entram arquivos que mudaram desde a cópia anterior. "
            "Hashes e datas em `snapshots/manifesto.csv`.")
    assets = [str(pasta / r["asset"]) for r in linhas]
    return [gh, "release", "create", tag, *assets, "--title", f"Dados brutos da ANEEL ({linhas[0]['data_utc']} UTC)",
            "--notes", nota, "--latest=false"]


def publicar(pasta: Path, manifesto: Path, dry_run: bool = False, gh: str = "gh") -> list[str]:
    """Cria a release com os arquivos preparados e, só se der certo, acrescenta as linhas ao manifesto.

    Com ``dry_run`` apenas devolve o comando que seria executado.
    """
    linhas = ler_manifesto(Path(pasta) / LOTE)
    if not linhas:
        raise ValueError(f"{pasta / LOTE} vazio ou ausente")
    cmd = comando_release(Path(pasta), linhas, gh)
    if dry_run:
        return cmd
    subprocess.run(cmd, check=True)
    _gravar_csv(manifesto, linhas, anexar=True)
    (Path(pasta) / LOTE).rename(Path(pasta) / (LOTE + ".publicado"))
    return cmd
