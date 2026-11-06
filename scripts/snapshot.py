"""Cópia datada dos dados brutos da ANEEL (a preparação é feita por scripts/robo.py; ver contrisk.snapshot).

    python scripts/snapshot.py publicar             # gh release create + acrescenta ao snapshots/manifesto.csv
    python scripts/snapshot.py publicar --dry-run   # só mostra o comando e as linhas do manifesto
    python scripts/snapshot.py resumo               # tamanho acumulado do que já foi guardado, por ano

``publicar`` precisa da CLI ``gh`` autenticada (no workflow, ``GH_TOKEN``). O manifesto só é alterado depois de
a release ser criada com sucesso.
"""
import argparse
import logging
import sys
from pathlib import Path

from contrisk import snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS, MANIFESTO = ROOT / "data" / "snapshots", ROOT / "snapshots" / "manifesto.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("snapshot")


def publicar(dry_run: bool, gh: str = "gh") -> int:
    pasta = snapshot.pasta_pendente(SNAPSHOTS)
    if pasta is None:
        log.info("nada a publicar: nenhuma pasta preparada em %s", SNAPSHOTS)
        return 0
    cmd = snapshot.publicar(pasta, MANIFESTO, dry_run=dry_run, gh=gh)
    linhas = snapshot.ler_manifesto(pasta / snapshot.LOTE) or snapshot.ler_manifesto(
        pasta / (snapshot.LOTE + ".publicado"))
    mb = sum(int(r["bytes_asset"]) for r in linhas) / 1e6
    if dry_run:
        print("dry-run: o comando que seria executado:")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        print(f"dry-run: {len(linhas)} arquivo(s), {mb:.1f} MB; linhas que entrariam em {MANIFESTO.name}:")
        for r in linhas:
            print("  " + ",".join(r[c] for c in snapshot.COLUNAS))
    else:
        log.info("release %s criada com %d arquivo(s), %.1f MB; manifesto atualizado", linhas[0]["tag"], len(linhas), mb)
    return 0


def resumo() -> int:
    linhas = snapshot.ler_manifesto(MANIFESTO)
    if not linhas:
        print("manifesto vazio")
        return 0
    por_ano: dict[str, list[int]] = {}
    for r in linhas:
        por_ano.setdefault(r["data_utc"][:4], []).append(int(r["bytes_asset"]))
    for ano, tam in sorted(por_ano.items()):
        print(f"{ano}: {len(tam)} arquivo(s) guardados, {sum(tam) / 1e6:.1f} MB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publicar")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gh", default="gh", help="executável da CLI do GitHub (para ensaios com um gh de mentira)")
    sub.add_parser("resumo")
    args = ap.parse_args()
    return publicar(args.dry_run, args.gh) if args.cmd == "publicar" else resumo()


if __name__ == "__main__":
    sys.exit(main())
