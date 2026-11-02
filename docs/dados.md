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

## O que não foi tratado

* Revisões posteriores dos dados: o backtest usa a base atual, não as versões publicadas em cada mês.
* Conjuntos que mudam de composição entre anos (desmembramento, fusão) são tratados pelo identificador
  `IdeConjUndConsumidoras`; quando o identificador muda, o histórico do novo conjunto começa do zero.
