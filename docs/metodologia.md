# Metodologia

O que é calculado, como e com que premissas. As decisões de tratamento dos dados (composição do DEC publicado,
anualização, compensações, limites, identificador do conjunto) estão em [`dados.md`](dados.md) e não são
repetidas aqui.

## 1. Perguntas

1. No mês *m* do ano *y*, qual é a probabilidade de cada conjunto fechar o ano com DEC (ou FEC) acima do limite
   anual da ANEEL?
2. Quanto de compensação esse conjunto deve pagar de *m+1* a dezembro?
3. Uma lista de alertas montada assim é melhor do que a regra que um analista usaria, a extrapolação do
   acumulado?

## 2. Dados e período

| Dado | Período | Granularidade |
|---|---|---|
| DEC, FEC e consumidores (indicadores coletivos de continuidade, ANEEL) | 2011 a 2026 (último mês publicado) | conjunto × mês |
| Compensações pagas (indicadores `PG*`, ANEEL) | 2011 a 2026 | conjunto × mês |
| Limites anuais de DEC e FEC (ANEEL) | por ano | conjunto × ano |

São cerca de 3.200 conjuntos de todas as distribuidoras. Os dados públicos chegam com atraso: na execução de
02/11/2026, iam até junho, julho ou agosto, conforme o conjunto ([`reports/status.json`](../reports/status.json)).

## 3. Pipeline

| Passo | Código | Saída |
|---|---|---|
| Download dos arquivos do CKAN da ANEEL | `contrisk.download.download` | `data/raw/` |
| Base mensal: DEC, FEC, consumidores, compensações e limites | `contrisk.build.base_mensal` | `data/interim/base_mensal.parquet` |
| Painel conjunto × ano × mês de corte, com features e alvos | `contrisk.features.painel` | `data/interim/painel.parquet` |
| Backtest 2021-2025 (cortes em março, junho e setembro) | `contrisk.model.backtest` | `reports/backtest_*.csv`, `data/interim/bt_prev.parquet` |
| Previsão do ano corrente | `contrisk.model.prever` | `reports/previsao_<ano>.csv` |
| Relatórios, figuras, resumo e Power BI | `contrisk.report` | `reports/`, `powerbi/` |
| Indicadores de decisão (alarmes falsos, conjuntos acionáveis, R$) | `scripts/resumo_decisao.py` | `reports/resumo_decisao*.csv` |

Tudo, exceto o último passo, é orquestrado por `scripts/robo.py`, que o workflow mensal executa.

## 4. Definições e fórmulas

**Indicador anual:** soma dos 12 meses publicados, `DEC_y = Σ_{mês=1..12} DEC_mês`. Só entram como alvo os anos
com os 12 meses.

**Violação:** `DEC_y > limite_DEC_y` (idem para o FEC). Limite ausente ou zero deixa o alvo indefinido.

**Features no corte *m* do ano *y*** (`features.painel`). Todas usam só os meses 1..*m* de *y* e os anos
anteriores completos:

| Feature | Fórmula |
|---|---|
| Acumulado | `DEC_acum = Σ_{mês=1..m} DEC_mês` |
| Últimos 3 meses | soma de *m-2* a *m* |
| Projeção linear | `DEC_acum × 12 / m` |
| Fração sazonal histórica | média, nos 5 anos anteriores, de `DEC_acum(m) / DEC_anual` no mesmo conjunto |
| Projeção sazonal | `DEC_acum / max(fração_sazonal, 0,05)` |
| Projeção pelo restante histórico | `DEC_acum + média do DEC de m+1 a dezembro nos 5 anos anteriores` |
| Razões ao limite | cada projeção, o acumulado e os fechamentos de *y-1* e *y-2* divididos pelo limite de *y* |
| Folga | `limite − DEC_acum` (negativa: o conjunto já passou do limite) |
| Histórico | fechamento, limite e violação de *y-1* e *y-2*; variação do limite |
| Compensações | pago no ano até *m*, últimos 3 meses e restante histórico do ano |
| Porte e distribuidora | consumidores; distribuidora como variável categórica |

**Modelos** (`model.py`):

* P(violar DEC) e P(violar FEC): LightGBM binário, 400 árvores, taxa de aprendizado 0,05, 31 folhas, mínimo de
  100 amostras por folha. Na previsão do ano corrente, a média de 3 sementes.
* R$ de compensação de *m+1* a dezembro: LightGBM com objetivo Tweedie (potência 1,3), adequado a valores com
  muitos zeros e cauda longa.
* O treino usa todos os meses de corte (1 a 11) dos anos anteriores ao ano previsto, então o mesmo modelo
  atende qualquer mês.

**Agregados do ano corrente:**

* Violações esperadas: `Σ P(violar)` sobre os conjuntos. Por linearidade da esperança, é a contagem esperada se as
  probabilidades forem calibradas, seja qual for a dependência entre DEC e FEC.
* `P(violar DEC ou FEC) = P_DEC + P_FEC − P(DEC e FEC)`, com `P(DEC e FEC) = Φ2(Φ⁻¹(P_DEC), Φ⁻¹(P_FEC); ρ)`
  (cópula gaussiana). `ρ` é estimado por máxima verossimilhança das quatro combinações (viola ou não DEC, viola ou
  não FEC) nas previsões fora da amostra do backtest (seção 5). Com `ρ = 0` vale a fórmula da independência,
  `1 − (1 − P_DEC)(1 − P_FEC)`.
* Risco alto: `P(violar DEC ou FEC) ≥ 0,5`.

## 5. Desenho de validação

**Origem móvel.** Para cada ano de teste *t* de 2021 a 2025, o modelo é treinado só com anos < *t* e avaliado nos
cortes de março, junho e setembro de *t*. O teste `test_painel_usa_so_meses_ate_o_corte_e_anos_anteriores`
coloca um evento enorme em novembro e confere que ele não aparece nas features de junho.

**Dependência entre DEC e FEC** (`model.backtest_dependencia`). Para cada ano de teste *t*, `ρ` é estimado só com
as previsões fora da amostra dos anos de 2016 a *t-1*. O backtest roda de 2016 a 2025, mas os anos de 2016 a 2020
servem só para calibrar o `ρ` de 2021: as métricas dos modelos continuam sendo as de 2021 a 2025. Independência e
cópula são comparadas sobre as mesmas probabilidades de DEC e de FEC (Brier e ECE de `P(DEC ou FEC)`, log-verossimilhança
conjunta, contagens esperadas contra observadas). O critério de adoção foi fixado antes de rodar: a cópula vira o
padrão se o Brier for menor em pelo menos 4 dos 5 anos e pelo menos 1% menor no agregado, e se a log-verossimilhança
conjunta agregada for maior. O resultado e a decisão ficam em `reports/dependencia_dec_fec.json`, que o robô mensal lê.

**Baselines** (a decisão é "sinalizar se a razão ao limite > 1"):

| Regra | Escore | O que representa |
|---|---|---|
| Extrapolação linear | `DEC_acum × 12/m ÷ limite` | a conta que se faz numa planilha de acompanhamento |
| Fração sazonal histórica | `DEC_acum ÷ fração_sazonal ÷ limite` | a planilha com sazonalidade |
| Violou no ano passado | `DEC_{y-1} ÷ limite` | "quem violou vai violar de novo" |

Para comparar calibração, o escore de cada regra é convertido em probabilidade por uma regressão logística
ajustada só nos anos de treino, separadamente por mês de corte.

**Métricas:**

| Métrica | Definição | Leitura |
|---|---|---|
| AUC | probabilidade de um conjunto que violou ter escore maior que um que não violou | ordenação da lista inteira |
| Brier | `média((p − y)²)` | erro da probabilidade; menor é melhor |
| ECE | média ponderada de `|taxa observada − p média|` em 10 faixas de probabilidade | calibração: "70% quer dizer 70%?" |
| Precisão | acertos / sinalizados | quantos alertas se confirmam |
| Recall | acertos / violações | quantas violações a lista pega |
| R$ nos sinalizados | compensação paga de *m+1* a dezembro nos conjuntos sinalizados | quanto da conta está na lista |

**Indicadores de decisão** (`scripts/resumo_decisao.py`, média por ano de 2021 a 2025): conjuntos sinalizados,
alarmes falsos, violações perdidas, sinalizados que ainda estavam dentro do limite no corte e R$ do restante do
ano nos sinalizados.

## 6. Premissas e parâmetros

| Premissa | Valor | Por quê | Efeito se mudar |
|---|---|---|---|
| Limiar de alerta | P ≥ 0,5 | ponto neutro; ajustável no painel | mais baixo: pega mais violações e dá mais alarmes falsos |
| Histórico do perfil sazonal | 5 anos | equilíbrio entre estabilidade e mudança da rede | |
| Cortes avaliados | março, junho e setembro | início, meio e fim do período útil para agir | |
| Dependência DEC × FEC | cópula gaussiana, `ρ` ≈ 0,60 estimado no backtest | os dois eventos andam juntos (correlação de 0,58 a 0,60 depois de descontar o que os modelos explicam) | `ρ = 0` (independência) superestima `P(DEC ou FEC)`; a cópula reduziu o Brier em 1,2% |
| Compensação no mês informado | sim | é como a ANEEL publica | apurações trimestrais e anuais concentram valores em certos meses |

## 7. Como interpretar os resultados

* **Probabilidade de um conjunto:** é a chance de fechar o ano acima do limite com o que se sabe hoje. Na faixa de
  40% a 85%, o modelo superestima o risco em cerca de 5 a 10 pontos (figura de calibração), então trate esses
  valores como teto.
* **Precisão e recall:** o modelo troca um pouco de recall por bem mais precisão, sobretudo cedo no ano. Ele dá
  menos alarmes falsos e deixa passar mais violações que a extrapolação. Se o custo de deixar passar é muito maior
  que o de inspecionar, baixe o limiar.
* **Conjuntos sinalizados ainda dentro do limite:** são os que importam para a ação. Quem já passou do limite
  dificilmente volta.
* **R$ esperado:** no backtest, o total do país foi subestimado em cerca de 15%. Leia como piso.
* **Violações esperadas:** a soma de probabilidades é uma estimativa do número de violações, não uma lista.
