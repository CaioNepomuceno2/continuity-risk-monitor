# Decisões de dados

## Fontes

Dataset `indicadores-coletivos-de-continuidade-dec-e-fec` do portal de dados abertos da ANEEL (CKAN), baixado
por [`src/contrisk/download.py`](../src/contrisk/download.py):

| Arquivo | Uso |
|---|---|
| `indicadores-continuidade-coletivos-2010-2019.parquet` e `-2020-2029.parquet` | DEC, FEC e número de consumidores (`NumCon`) por conjunto e mês |
| `indicadores-continuidade-coletivos-compensacao-2010-2019.parquet` e `-2020-2029.parquet` | R$ pagos em compensações por conjunto e mês |
| `indicadores-continuidade-coletivos-limite.csv` | limites anuais de DEC e FEC por conjunto |
| `dominio-indicadores.csv` | dicionário das siglas de indicadores |

Os arquivos de 2020-2029 e de limites mudam todo mês; o robô sempre os baixa de novo.

## DEC e FEC

* O `DEC` (e o `FEC`) publicado é o indicador apurado para comparação com o limite. Ele bate com
  `DECIP + DECIND` (interrupções de origem interna, programadas e não programadas, sem as expurgáveis) em
  **100% dos 37.584 conjuntos × mês de 2024** (diferença < 0,011 h).
* O indicador **anual é a soma dos 12 meses**. Anos com menos de 12 meses publicados não entram como alvo nem
  como fechamento de ano anterior.
* Só entram meses 1 a 12 (`NumPeriodoIndice`); registros trimestrais ou anuais, quando existem, ficam de fora.

## Limites

* Um valor de DEC e um de FEC por conjunto e ano (`AnoLimiteQualidade`). O limite de *y* é conhecido antes de
  *y* começar, então pode ser feature.
* Limite ausente ou zero: o alvo de violação fica indefinido (NaN) e a linha não entra no treino nem na
  avaliação daquele indicador. Em 2026 há limites para 3.183 conjuntos.

## Compensações

* Soma de todos os indicadores `PG*` (R$ pagos por violação de DIC, FIC, DMIC, DICRI e DISE, nas apurações
  mensal, trimestral e anual), no mês em que foram informados.
* O alvo de R$ é o pago de *m+1* a dezembro do ano *y*.
* A compensação é por consumidor, não por conjunto. A relação com o limite coletivo é só empírica:
  [`reports/compensacao_vs_violacao.csv`](../reports/compensacao_vs_violacao.csv) mostra o R$ médio por faixa de
  DEC anual / limite em 2021-2025.

## Corte temporal

Para o ano *y* e o mês de corte *m* ([`src/contrisk/features.py`](../src/contrisk/features.py)):

* entram os meses 1..*m* de *y* e os anos anteriores completos;
* o perfil sazonal do conjunto (fração do ano realizada até *m*, DEC do restante do ano, R$ do restante do
  ano) é a média dos 5 anos anteriores a *y*;
* no backtest, o modelo do ano *t* é treinado só com anos < *t* (todos os meses de corte de 1 a 11).

O teste `test_painel_usa_so_meses_ate_o_corte_e_anos_anteriores` coloca um evento enorme em novembro e
confere que ele não aparece nas features de junho do mesmo ano.

## Cópia datada dos downloads brutos

A ANEEL regenera os arquivos com frequência: em 06/11/2026, os cinco arquivos grandes tinham sido gerados de novo
em 05/11 (entre 08:33 e 16:58 GMT), no mesmo dia da execução do robô. Meses passados também podem ser revisados
pelas distribuidoras. Para formar, com o tempo, a base "como foi publicada", o robô guarda uma cópia datada do que
baixou ([`src/contrisk/snapshot.py`](../src/contrisk/snapshot.py)).

**Tamanho de um ciclo mensal** (download completo no GitHub Actions, que começa sem cache; medido em 06/11/2026,
4 min 50 s nesta máquina):

| Arquivo | Baixado | Guardado | Como |
|---|---|---|---|
| `indicadores-continuidade-coletivos-2010-2019.parquet` | 97,5 MB | 97,5 MB | idêntico |
| `indicadores-continuidade-coletivos-2020-2029.parquet` | 58,8 MB | 58,8 MB | idêntico |
| `indicadores-continuidade-coletivos-compensacao-2010-2019.parquet` | 104,9 MB | 104,9 MB | idêntico |
| `indicadores-continuidade-coletivos-compensacao-2020-2029.parquet` | 116,1 MB | 116,1 MB | idêntico |
| `indicadores-continuidade-coletivos-limite.csv` | 52,4 MB | 4,2 MB | gzip |
| `dominio-indicadores.csv` | 0,05 MB | 0,006 MB | gzip |
| **Total** | **429,7 MB** | **381,5 MB** | |

**Escolha: *assets* de uma release do GitHub, com manifesto no git.**

* **Git não serve:** uma cópia completa é 80 vezes o repositório inteiro (hoje com cerca de 5 MB), e Parquet não
  tem delta no git.
* **Parquet fica como veio.** Ele já é comprimido: o gzip mediu 89% a 91% do tamanho original, e a cópia idêntica
  byte a byte permite conferir o sha256 contra o que a ANEEL serviu. Só o CSV vai em gzip (8% do tamanho).
* **Só entra o que mudou.** O robô compara o sha256 de cada arquivo com o último registro de
  [`snapshots/manifesto.csv`](../snapshots/manifesto.csv). Arquivo igual ao da cópia anterior não é guardado de novo.
* **O manifesto** (data, arquivo, bytes, sha256, data de modificação no CKAN, nome do *asset*) é versionado no git
  e ocupa cerca de 2 KB por mês. Os arquivos ficam nos *assets* de uma release por execução
  (`dados-aneel-AAAAMMDD-HHMM`), que não entram no histórico nem no tamanho do repositório.

**Tamanho esperado por ano** (cálculo com os números da tabela, não medição do ano):

| Cenário | Por mês | Por ano |
|---|---|---|
| Só os três arquivos que a ANEEL atualiza todo mês mudam (2020-2029 de indicadores e de compensações, e limites) | 179,1 MB | cerca de 2,1 GB |
| Todos os arquivos mudam todo mês | 381,5 MB | cerca de 4,6 GB |

Mais uma cópia inicial completa (381,5 MB) na primeira execução. Os arquivos de 2010-2019 podem ou não mudar de
um mês para o outro; o manifesto vai mostrar qual dos dois cenários vale, e `python scripts/snapshot.py resumo`
soma o que já foi guardado por ano.

**Como o fluxo se protege:**

* `scripts/robo.py` só prepara a cópia em `data/snapshots/` (fora do git); qualquer erro nessa etapa é registrado
  e o robô segue.
* O envio para a release é um passo próprio do workflow (`python scripts/snapshot.py publicar`), com
  `continue-on-error`, antes do passo que versiona os relatórios. Se falhar, os relatórios saem do mesmo jeito.
* O manifesto só é alterado depois de a release ser criada. Se o envio falhar, a cópia continua pendente e o mês
  seguinte tenta de novo.
* Ensaio sem envio: `python scripts/snapshot.py publicar --dry-run` mostra o comando e as linhas do manifesto.

**Como refazer uma base "como foi publicada":** baixe os *assets* da release da data desejada (e, para os arquivos
que não mudaram naquela data, os da release mais recente anterior que os contém, conforme o manifesto), descompacte
os `.csv.gz`, junte tudo numa pasta com os nomes originais e aponte `build.base_mensal` para ela.

## O que não foi tratado

* Revisões posteriores dos dados: o backtest usa a base atual, não as versões publicadas em cada mês. O robô
  passou a guardar uma cópia datada de cada download (seção abaixo), então a base "como foi publicada"
  começa na primeira cópia e só cresce daí em diante; **o backtest "como foi publicado" ainda não foi
  feito**, porque não há meses de cópias para isso.
* Conjuntos que mudam de composição entre anos (desmembramento, fusão) são tratados pelo identificador
  `IdeConjUndConsumidoras`; quando o identificador muda, o histórico do novo conjunto começa do zero.
