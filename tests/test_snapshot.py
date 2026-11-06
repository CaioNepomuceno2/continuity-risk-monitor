"""Testes da cópia datada dos dados brutos da ANEEL (sem rede, sem GitHub)."""
import gzip
import importlib.util
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests

from contrisk import snapshot as S
from contrisk.download import ARQUIVOS

T1 = datetime(2026, 12, 5, 9, 17, tzinfo=UTC)
T2 = datetime(2027, 1, 5, 9, 17, tzinfo=UTC)


def _raw(tmp_path, extra=""):
    """Um arquivo falso para cada recurso que o robô baixa (CSV de texto, o resto parecendo Parquet)."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for i, nome in enumerate(ARQUIVOS):
        conteudo = (b"PAR1" + bytes([i]) * 5000 + b"PAR1") if nome.endswith(".parquet") else (
            ("IdeConjUndConsumidoras;AnoLimiteQualidade\n" + "1;2026\n" * 3000).encode())
        (raw / nome).write_bytes(conteudo + extra.encode())
    return raw


def _gh_ok(monkeypatch):
    chamadas = []
    monkeypatch.setattr(S.subprocess, "run", lambda cmd, check: chamadas.append(cmd))
    return chamadas


META = {n: "2026-12-05T08:00:00" for n in ARQUIVOS}


def test_primeira_copia_leva_todos_os_arquivos_com_hash_e_sem_alterar_o_parquet(tmp_path):
    raw, saida, manifesto = _raw(tmp_path), tmp_path / "saida", tmp_path / "manifesto.csv"
    pasta = S.preparar(raw, saida, manifesto, agora=T1, meta=META)
    assert pasta.name == "dados-aneel-20261205-0917"
    linhas = S.ler_manifesto(pasta / S.LOTE)
    assert [r["arquivo"] for r in linhas] == ARQUIVOS
    for r in linhas:
        original = (raw / r["arquivo"]).read_bytes()
        assert r["sha256"] == S.sha256(raw / r["arquivo"]) and int(r["bytes_originais"]) == len(original)
        assert r["ckan_last_modified"] == "2026-12-05T08:00:00" and r["tag"] == pasta.name
        if r["arquivo"].endswith(".parquet"):
            assert r["asset"] == r["arquivo"] and (pasta / r["asset"]).read_bytes() == original   # idêntico
        else:
            assert r["asset"] == r["arquivo"] + ".gz"
            assert gzip.decompress((pasta / r["asset"]).read_bytes()) == original                 # CSV em gzip
            assert int(r["bytes_asset"]) < len(original) / 5
    assert not manifesto.exists()                      # preparar não toca no manifesto


def test_so_o_que_mudou_desde_o_ultimo_registro_e_copiado(tmp_path, monkeypatch):
    raw, saida, manifesto = _raw(tmp_path), tmp_path / "saida", tmp_path / "manifesto.csv"
    _gh_ok(monkeypatch)
    S.publicar(S.preparar(raw, saida, manifesto, agora=T1, meta=META), manifesto)
    assert S.preparar(raw, saida, manifesto, agora=T2, meta=META) is None          # nada mudou: nada a guardar
    alterado = ARQUIVOS[1]
    (raw / alterado).write_bytes(b"PAR1 revisado PAR1")
    pasta = S.preparar(raw, saida, manifesto, agora=T2, meta=META)
    assert [r["arquivo"] for r in S.ler_manifesto(pasta / S.LOTE)] == [alterado]
    assert sorted(p.name for p in pasta.iterdir()) == sorted([alterado, S.LOTE])


def test_arquivo_ausente_e_ignorado_e_sem_nenhum_arquivo_nao_ha_o_que_guardar(tmp_path):
    raw = _raw(tmp_path)
    (raw / ARQUIVOS[0]).unlink()
    pasta = S.preparar(raw, tmp_path / "saida", tmp_path / "m.csv", agora=T1, meta=META)
    assert ARQUIVOS[0] not in [r["arquivo"] for r in S.ler_manifesto(pasta / S.LOTE)]
    vazio = tmp_path / "vazio"
    vazio.mkdir()
    assert S.preparar(vazio, tmp_path / "saida2", tmp_path / "m.csv", agora=T1, meta=META) is None


def test_publicar_em_dry_run_so_monta_o_comando(tmp_path, monkeypatch):
    raw, saida, manifesto = _raw(tmp_path), tmp_path / "saida", tmp_path / "manifesto.csv"
    pasta = S.preparar(raw, saida, manifesto, agora=T1, meta=META)

    def proibido(*a, **k):
        raise AssertionError("dry-run não pode chamar o gh")

    monkeypatch.setattr(S.subprocess, "run", proibido)
    cmd = S.publicar(pasta, manifesto, dry_run=True)
    assert cmd[:4] == ["gh", "release", "create", pasta.name] and "--latest=false" in cmd
    assert sum(c.endswith((".parquet", ".csv.gz")) for c in cmd) == len(ARQUIVOS)
    assert not manifesto.exists() and (pasta / S.LOTE).exists()


def test_publicar_so_grava_o_manifesto_depois_de_a_release_dar_certo(tmp_path, monkeypatch):
    raw, saida, manifesto = _raw(tmp_path), tmp_path / "saida", tmp_path / "manifesto.csv"
    pasta = S.preparar(raw, saida, manifesto, agora=T1, meta=META)

    def falha(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(S.subprocess, "run", falha)
    with pytest.raises(subprocess.CalledProcessError):
        S.publicar(pasta, manifesto)
    assert not manifesto.exists() and S.pasta_pendente(saida) == pasta            # pode tentar de novo
    chamadas = _gh_ok(monkeypatch)
    S.publicar(pasta, manifesto)
    assert len(chamadas) == 1 and len(S.ler_manifesto(manifesto)) == len(ARQUIVOS)
    assert S.pasta_pendente(saida) is None                                        # lote marcado como publicado
    # o manifesto acumula: uma segunda release acrescenta, não reescreve
    (raw / ARQUIVOS[2]).write_bytes(b"PAR1 novo PAR1")
    S.publicar(S.preparar(raw, saida, manifesto, agora=T2, meta=META), manifesto)
    linhas = S.ler_manifesto(manifesto)
    assert len(linhas) == len(ARQUIVOS) + 1 and linhas[-1]["arquivo"] == ARQUIVOS[2]
    assert manifesto.read_text(encoding="utf-8").splitlines()[0] == ",".join(S.COLUNAS)


def test_preparar_sem_falhar_nunca_derruba_o_robo(tmp_path, monkeypatch, caplog):
    def quebra(*a, **k):
        raise OSError("disco cheio")

    monkeypatch.setattr(S, "preparar", quebra)
    assert S.preparar_sem_falhar(tmp_path, tmp_path / "s", tmp_path / "m.csv") is None
    assert "disco cheio" in caplog.text


def test_metadados_do_ckan_sao_melhor_esforco(monkeypatch):
    def sem_rede(*a, **k):
        raise requests.ConnectionError("sem rede")

    monkeypatch.setattr(S.requests, "get", sem_rede)
    assert S.metadados_ckan() == {}


def test_script_publicar_em_dry_run_mostra_comando_e_linhas(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("snapshot_cli", Path(__file__).parents[1] / "scripts" / "snapshot.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "SNAPSHOTS", tmp_path / "saida")
    monkeypatch.setattr(cli, "MANIFESTO", tmp_path / "manifesto.csv")
    assert cli.publicar(dry_run=True) == 0                                         # nada preparado: sai limpo
    S.preparar(_raw(tmp_path), tmp_path / "saida", tmp_path / "manifesto.csv", agora=T1, meta=META)
    assert cli.publicar(dry_run=True) == 0
    saida = capsys.readouterr().out
    assert "gh release create dados-aneel-20261205-0917" in saida and "linhas que entrariam" in saida
    assert not (tmp_path / "manifesto.csv").exists()
