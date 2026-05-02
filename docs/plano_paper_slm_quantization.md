# Plano de Paper: Quantização End-to-End para Deployment de SLMs em Edge

> **Status**: Versão 3.0 (pré-execução) — incorpora todas as correções e adições resultantes da leitura crítica completa do MLWQ (Seções 4, 6, 7) e do TurboQuant (Seções 1, 2, 3). Esta é a versão final do plano antes de iniciar execução. Mudanças posteriores devem ser orientadas por evidência empírica, não por leitura adicional.

> **Histórico de versões**:
> - V1: plano inicial com sobreposição de QJL/PolarQuant/TurboQuant
> - V2.0: TurboQuant como método principal, PolarQuant descartado, alocador formalizado, Jetson como target
> - V2.1: leitura crítica Seção 4 do MLWQ (motivação empírica frágil)
> - V2.2: leitura crítica Seção 6 do MLWQ (WM/RM ratio, MMLU/GSM8K, risco de regressão)
> - V2.3: leitura crítica Seção 7 do MLWQ (cenário-alvo, variantes hardware-aligned, ablação A8)
> - V3.0: leitura crítica TurboQuant §§1-3 (gap vs SLB, Hadamard randomizada, Lloyd-Max Beta vs Gaussian, custo de Π, norma fora do regime, variante two-stage radical)

> **Marcadores de adição**: pesquisar por `[ADIÇÃO V2]`, `[ADIÇÃO V2.3]`, `[ADIÇÃO V3]` para localizar mudanças por época.

---

## 1. Síntese Executiva

### Tese central

Métodos weight-only de quantização para SLMs (como MLWQ) são insuficientes para deployment em contexto longo, pois a memória dinâmica (KV cache) cresce linearmente com a sequência e passa a dominar o consumo total de memória, anulando o ganho da compressão dos pesos.

### Contribuição principal

Propomos um framework de **quantização end-to-end com alocação global de bits via otimização Lagrangiana**, combinando:

- Quantização multi-nível dos pesos (base MLWQ, com substituições baseadas em TurboQuant)
- Quantização online da KV cache via TurboQuant
- **Alocador global de bits formalizado** que distribui orçamento entre pesos e KV cache conforme contexto, sensibilidade de camada e restrições de hardware edge
- Validação em hardware edge real (não apenas datacenter GPU)

### Diferencial em relação ao MLWQ

| Eixo | MLWQ | Nosso paper |
|------|------|-------------|
| Escopo de compressão | Apenas pesos | Pesos + KV cache + ativações |
| Fundamentação teórica | Heurística empírica | Lower bound informacional + bound de distorção |
| Alocação de bits | Enumeração 2^l por bloco | Water-filling Lagrangiano analítico |
| Hardware avaliado | RTX 4090 24GB | Jetson Orin Nano + RTX 4090 |
| Long-context | Não avaliado | Eixo principal (até 16K-32K tokens) |
| Métricas | PPL, zero-shot | PPL, zero-shot, VRAM, tokens/s, prefill, decode, LongBench, NIH |

---

## 2. Análise dos Artigos de Suporte

### 2.1 TurboQuant (Zandieh et al., abr/2025) — **base técnica principal (~70% das ferramentas)**

**Ferramentas que usaremos**:

- Random rotation antes de quantização (Algorithm 1, linha 2) — substitui o grouping Hessiano caro do MQSA
- Lloyd-Max sobre distribuição Beta induzida (Eq. 4) — substitui o quantizador uniforme do MLWQ
- Lower bound via Yao + Shannon (Theorem 3) — fundamenta teoricamente a alocação de bits
- Bound √(3π)/2 · 4⁻ᵇ (Theorem 1) — vira a função objetivo do water-filling
- Pipeline de duas etapas (MSE quantizer + QJL no resíduo) — para inner-product unbiased em KV cache

**Limitação a tratar**: avaliado em A100, não em edge. Custo da rotação aleatória precisa ser medido honestamente em hardware restrito.

### 2.2 QJL (Zandieh et al., jun/2024) — **componente cirúrgico (~20%)**

**Uso**: aparece dentro do TurboQuant_prod como o "1-bit no resíduo". Não é método rival do TurboQuant — é componente dele.

**Papel no paper**:

- Ablação ("o que perdemos se removermos a etapa MSE inicial e usarmos só QJL puro?")
- Baseline teórico para preservação de inner product

### 2.3 PolarQuant (Han et al., fev/2025) — **citação contextual apenas (~10%)**

**Crítica honesta**: prefill 4× mais lento que exato (Tabela 2 do paper original) é eliminatório para edge deployment. Recursão polar é hostil a vetorização.

**Papel no paper**:

- Citação histórica como antecessor do TurboQuant
- Possível baseline negativo se houver tempo
- **Não usar como variante operacional** (correção em relação ao plano anterior)

### 2.4 MLWQ (Hu et al., EMNLP 2025) — **paper-base**

**Componentes a preservar**:

- Conceito de mixed-precision inter-camada (BPLL) — mas reformulado via water-filling
- Métrica de loss inter-camada channel-wise (Eq. 3) — útil como sinal complementar
- Foco em SLMs e edge deployment como nicho

**Componentes a substituir**:

- Quantizador uniforme (Eq. 1) → Lloyd-Max sobre distribuição rotacionada
- Grouping Hessiano por salience (MQSA) → random rotation que destrói outliers
- Enumeração 2^l do BPLL → water-filling analítico
- Cobertura apenas de pesos → extensão para KV cache

<!-- [ADIÇÃO V2 — leitura crítica da Seção 4 do MLWQ] -->

#### 2.4.1 Fraquezas metodológicas da motivação empírica do MLWQ

A Seção 4 do MLWQ (Motivation) sustenta as duas decisões centrais de design (BPLL e MQSA) com evidência empírica que, sob escrutínio, é mais frágil do que parece à primeira leitura. Mapear essas fraquezas é importante porque várias delas viram diferenciais defensáveis no nosso paper:

- **Observation 1 baseada em amostra ínfima**: a Figura 1 do MLWQ mostra loss por camada apenas para o **bloco 3 do OPT-350M**, sob 2-bit e 4-bit. Não há evidência apresentada de que o padrão de heterogeneidade inter-camada se mantém em outros blocos do mesmo modelo, em outros modelos (Llama-3.2, Phi, SmolLM2), ou em outros bit-widths. Os autores assumem generalidade implicitamente.

- **Decomposição implícita da Observation 1**: a observação combina dois fatos distintos sob um único enunciado — (Fato A) heterogeneidade inter-camada de sensibilidade, e (Fato B) monotonicidade de loss vs bit-width. O Fato A motiva o BPLL; o Fato B é precondição para a otimização da Eq. 4 funcionar. A estabilidade do *ranking* de sensibilidade entre bit-widths (que é o que faz BPLL ser tratável como otimização única) não é formalmente verificada.

- **Observation 2 sustentada por gráficos 3D ilustrativos**: as Figuras 2 e 3 do MLWQ mostram saliência via plots 3D de difícil leitura. O argumento de "distribuição não-bimodal com região intermediária" seria muito mais convincente com **histogramas 1D em escala log** da saliência por canal. A ausência dessa figura é furo metodológico relevante.

- **Escolha de n=3 grupos sem fundamentação teórica**: a Figura 4 mostra diminishing returns com cotovelo em n=3, mas os autores admitem explicitamente que valores maiores reduziriam ainda mais a perplexidade. A escolha é pragmática (trade-off com complexidade de bookkeeping/CUDA), não principled. Não há análise information-theoretic ou distributional que justifique três grupos especificamente.

- **Métrica Dist (Eq. 3) com premissa implícita**: a métrica de loss inter-camada baseia-se em diferenças de (μ, σ) por canal de ativação. Isso assume implicitamente que ativações são aproximadamente Gaussianas, caso contrário (μ, σ) não capturam a distribuição. Em camadas com outliers fortes (justamente as mais sensíveis à quantização, conforme literatura de Dettmers et al.), essa premissa quebra. MLWQ pode estar usando métrica que **subestima a sensibilidade exatamente nas camadas onde quantização mais dói**.

- **Limiares H e L não-especificados**: o MQSA seleciona top-H canais salientes e bottom-L não-salientes, mas não há fórmula explícita para H e L. Os autores afirmam que "dependem do bit-width inter-camada" sem fornecer mapeamento. Furo de reprodutibilidade.

**Implicação para o nosso paper**: cada uma dessas fraquezas é oportunidade de diferencial. Em particular, a substituição do MQSA por random rotation no nosso método **provavelmente refuta a Observation 2 empiricamente** — pós-rotação, a saliência se aproxima de Gaussian e a estrutura trimodal desaparece. Isso é resultado científico genuíno que vai além de "melhoramos métricas".

<!-- [/ADIÇÃO V2] -->

<!-- [ADIÇÃO V3 — postura informacional híbrida] -->

### 2.5 Postura informacional do método: híbrida explícita

A literatura de quantização divide-se em dois paradigmas com posturas informacionais opostas:

**Paradigma data-dependent (offline)**: MLWQ, AWQ, GPTQ, OmniQuant, OWQ. Calibração com amostras representativas, exploração de estrutura empírica dos pesos (saliência Hessiana, distribuição de outliers por canal), bit allocation guiada por dados. Vantagem: explora informação disponível. Desvantagem: assume que dados de calibração são representativos do uso em produção, sem garantia teórica.

**Paradigma data-oblivious (online)**: TurboQuant, QJL, PolarQuant. Random rotation ou JL transform aplicados sem ver dados, quantizadores universais via Lloyd-Max sobre distribuição induzida (Beta), garantias teóricas adversariais via Yao + Shannon Lower Bound. Vantagem: garantias informacionais formais válidas para qualquer distribuição. Desvantagem: sub-explora estrutura conhecida dos dados.

**Postura do nosso método**: **híbrida explícita**, separando os dois componentes:

- **Pesos (estático, conhecido a priori)** → data-dependent leve. Usamos calibração para o water-filling Lagrangiano (medir sensibilidade por camada via Eq. 3 do MLWQ adaptada), mas o quantizador escalar pós-rotação é universal (Lloyd-Max sobre distribuição Beta induzida, codebook offline). Não usamos Hessiana cara como MLWQ original.
- **KV cache (dinâmico, gerado online)** → data-oblivious estrito. TurboQuant_prod sem qualquer calibração — funciona em distribuição arbitrária por construção via Lemma 1.

**Por que essa escolha é defensável**:
1. Pesos têm estrutura conhecida e estável que vale explorar (sensibilidade inter-camada não muda durante inferência)
2. KV cache é gerado token a token, nunca está disponível antes — calibração é fundamentalmente impossível
3. A postura híbrida permite herdar lower bound informacional do TurboQuant para o componente KV (onde data-oblivious é necessário) sem abrir mão de eficiência prática nos pesos (onde data-dependent é viável)

**Contra-argumento esperado de revisor**: "Por que não data-oblivious puro também para pesos, herdando lower bound completo?". Resposta: porque pesos têm sensibilidade inter-camada heterogênea (Observation 1 do MLWQ é generalizável), e ignorar essa estrutura desperdiça bit-budget. O ganho prático supera a elegância teórica de uniformidade. Este trade-off é articulado explicitamente, não escondido.

<!-- [/ADIÇÃO V3] -->

---

## 3. Posicionamento do Problema

### 3.1 Pergunta de pesquisa principal

Como reduzir o custo total de inferência de SLMs (memória de pesos + memória de KV cache + latência de decode) em hardware edge, mantendo qualidade dentro de limite estrito, em regimes de contexto longo?

### 3.2 Perguntas secundárias

- **RQ1**: A partir de qual comprimento de contexto a KV cache passa a dominar a memória total em SLMs (135M–3B parâmetros)?
- **RQ2**: Quantizar pesos + KV cache com alocação global supera quantização independente com bit-budgets fixos?
- **RQ3**: O custo da random rotation do TurboQuant é absorvível em hardware edge com FFT-like Hadamard?
- **RQ4**: O método mantém qualidade em out-of-domain calibration (relevante para edge real)?

### 3.3 Hipóteses falsificáveis

- **H0** (nula): MLWQ isolado é suficiente; quantizar KV cache não traz ganho relevante mantendo qualidade.
- **H1**: MLWQ + TurboQuant para KV cache reduz peak VRAM em ≥ 30% em contexto ≥ 8K, com queda de PPL ≤ 3%.
- **H2**: Alocador global Lagrangiano supera bit-budgets fixos em ≥ 1.5% de PPL com mesmo orçamento de memória.
- **H3**: O ganho de KV cache quantization cresce monotonicamente com o contexto.
- **H4**: Em Jetson Orin Nano, o método mantém ≥ 85% do throughput de decoding em FP16 com ≥ 4× redução de memória total.

> **Critério de honestidade científica**: se H1 falhar em SmolLM2-135M mas suceder em Llama-3.2-1B, reportamos ambos. Se KV int4 simples atingir performance comparável a TurboQuant, reportamos isso e o paper foca no alocador global como contribuição principal.

---

## 4. Método Proposto

### 4.1 Visão geral

```
SLM original (FP16)
        ↓
Calibração com 128 amostras (WikiText2 + out-of-domain)
        ↓
[ESTÁTICO] Compressão dos pesos
  - Random Hadamard rotation por camada
  - Lloyd-Max sobre distribuição rotacionada
  - Mixed-precision inter-camada via water-filling
        ↓
[INFERÊNCIA] Compressão dinâmica da KV cache
  - TurboQuant_prod (MSE quantizer + QJL no resíduo)
  - Bit-budget por camada definido pelo alocador global
        ↓
Decoding com kernels CUDA otimizados (também Jetson)
```

<!-- [ADIÇÃO V2.3 — premissa explícita de cenário-alvo] -->

#### 4.1.1 Cenário-alvo e escopo de uso

O método é projetado especificamente para o seguinte cenário de deployment:

- **Decode autoregressivo (token por token)** — geração de texto streaming, não inferência batch
- **Batch size pequeno** (1–4 sequências simultâneas) — caso típico de assistente em edge
- **Memória como gargalo primário** — bandwidth de pesos e KV cache domina, não compute throughput
- **Hardware com memória limitada** — 4–8 GB de memória unificada (Jetson Orin Nano, smartphones com NPU)

**Cenários explicitamente fora do escopo**:

- **Prefill de prompts muito longos** (>16K tokens em uma única passada) — neste regime, ativações intermediárias dominam, e weight-only + KV cache quantization não é suficiente. Tratar prefill longo requer ativação quantization (SmoothQuant, QuaRot completo), que é problema mais difícil e fora do escopo deste trabalho.
- **Inferência server-side com batch grande** — neste regime, ativações dominam o consumo de memória e o método tem ganho marginal.

Esta delimitação é importante porque:
1. Justifica a escolha de quantizar apenas pesos + KV cache (não ativações)
2. Alinha as métricas reportadas (decode tokens/s, peak VRAM em decode) com o caso de uso real
3. Evita que revisor pergunte "por que não quantizam ativações?" — a resposta é "fora do escopo declarado"

<!-- [/ADIÇÃO V2.3] -->

### 4.2 Componente 1: Quantização de Pesos com Rotação

Substituição do MQSA do MLWQ. Para cada camada `l` com matriz de pesos `W_l ∈ R^(d_out × d_in)`:

1. Aplicar rotação Hadamard randomizada `Π_l` (custo O(d log d) via FFT, vs O(d²) da rotação densa)
2. Calcular pesos rotacionados `W̃_l = Π_l · W_l`
3. Aplicar Lloyd-Max precomputado para distribuição rotacionada (codebook offline, custo zero em inferência)
4. Armazenar índices + escala única por canal (sem zero-points por bloco — overhead eliminado)

**Justificativa teórica**: pós-rotação, distribuição de coordenadas converge para Gaussian em alta dimensão (TurboQuant Lemma 1). Lloyd-Max é provadamente ótimo escalar para essa distribuição. Outliers são destruídos por construção, eliminando necessidade do grouping caro do MQSA.

<!-- [ADIÇÃO V3 — especificação técnica da rotação] -->

#### 4.2.1 Especificação técnica: Hadamard randomizada

A escolha exata de Π afeta tanto teoria quanto custo. Três opções comparadas:

| Opção | Custo computacional | Custo de armazenamento | Distribuição induzida |
|-------|---------------------|------------------------|----------------------|
| Π Gaussian densa (TurboQuant original via QR) | O(d²) por aplicação, O(d³) para gerar | O(d²) — para d=4096, 64 MB em FP32 | Haar exata em O(d) |
| Hadamard pura (H determinística) | O(d log d) | O(1) — recursiva | Determinística, **não é Haar** |
| **Hadamard randomizada (D·H, nossa escolha)** | O(d log d) | O(d) — vetor de sinais ±1 | Aproxima Haar; bounds via Ailon-Chazelle (2009) |

**Decisão**: usamos `Π = D·H` onde D é matriz diagonal com sinais aleatórios `±1` i.i.d. e H é Hadamard determinística. Justificativa:

1. Custo `O(d log d)` é absorvível em edge (Jetson Orin Nano), enquanto `O(d²)` da rotação Gaussiana densa é inviável para d=3072 (Llama-3.2-3B).
2. Custo de armazenamento `O(d)` (vetor de sinais) vs `O(d²)` (matriz densa) — diferença entre KB e MB por camada.
3. Bounds teóricos para Lloyd-Max sobre distribuição induzida pela Hadamard randomizada são derivados em Ailon & Chazelle (Fast JL Transform, 2009). Vale notar: bounds são ligeiramente mais frouxos que Haar exata, mas mantêm o decaimento `4⁻ᵇ` com constante incrementada.

**Trade-off explícito**: ao usar Hadamard randomizada em vez de Haar exata, sacrificamos algum fator constante no bound (estimativa: 1.2× a 1.5× pior que TurboQuant original) em troca de viabilidade computacional em edge. Este trade-off é articulado no paper, não escondido.

#### 4.2.2 Lloyd-Max: aproximação Gaussian vs Beta exata

A distribuição induzida pelo Lemma 1 do TurboQuant é Beta com parâmetros `(d-3)/2`. Em alta dimensão `(d → ∞)` converge para `N(0, 1/d)`. Os centróides Lloyd-Max diferem entre as duas distribuições para d finito:

- **Aproximação Gaussian**: codebook fixo via tabelas clássicas (Max 1960). Funciona bem para d ≥ 1024 (Llama-3.2, Phi-2).
- **Beta exata**: solução numérica iterativa por dimensão. Necessário para d pequeno (SmolLM2-135M tem d=576).

**Decisão prática**: pré-computar dois conjuntos de codebooks (Gaussian para d ≥ 1024, Beta exata para d < 1024) e selecionar conforme modelo. Reportar gap entre os dois em ablação para validar que aproximação Gaussian é suficiente nos modelos principais.

#### 4.2.3 Norma fora do regime quantizado

Lemma 1 do TurboQuant assume vetores na esfera unitária. Pesos reais não estão normalizados. Solução padrão: armazenar norma `‖W_l[:, k]‖` por canal `k` em FP16 e quantizar `W_l[:, k] / ‖W_l[:, k]‖`.

**Contabilidade de bits**:
- Pesos quantizados: `b · d_in · d_out` bits por camada
- Normas FP16: `16 · d_in` bits por camada (uma norma por canal de saída)
- Overhead: `16/(b · d_out)` — para `b=3, d_out=4096`, isso é 0.13% do total

Desprezível em alta dimensão, mas precisa ser contabilizado honestamente nas métricas de compressão (não inflar ratio ignorando normas).

<!-- [/ADIÇÃO V3] -->

<!-- [ADIÇÃO V2 — leitura crítica da Seção 4 do MLWQ] -->

> **Nota de contribuição empírica**: a aplicação da rotação Hadamard nos pesos provavelmente **refuta empiricamente a Observation 2 do MLWQ** (de que saliência intra-camada exibe estrutura trimodal salient/ordinary/non-salient). Pós-rotação, espera-se que a distribuição de saliência se aproxime de Gaussian unimodal, com canais previamente "salientes" deixando de ser concentrados em coordenadas específicas. Isso significa que substituímos uma heurística trimodal (MQSA com n=3 grupos) por quantizador escalar uniforme sobre distribuição rotacionada — mais simples, com garantia teórica via TurboQuant Theorem 1, e potencialmente refutando uma das motivações empíricas centrais do paper-base. Esta verificação é experimento dedicado na Fase 5 (ver A7).

<!-- [/ADIÇÃO V2] -->

### 4.3 Componente 2: Quantização Online da KV Cache

Para cada token gerado, aplicar TurboQuant_prod sobre `k_t` e `v_t`:

1. **Etapa MSE**: quantizar com b-1 bits via Lloyd-Max após rotação
2. **Etapa QJL**: aplicar 1-bit JL no resíduo `r = k_t - dequantize(quantize(k_t))`
3. **Reconstrução**: `k̂_t = dequantize_mse + ‖r‖ · QJL_dequantize`

**Garantia**: estimador de inner product `⟨q_t, k̂_t⟩` é unbiased (TurboQuant Theorem 2), com distorção ≤ √(3π)/2 · ‖q‖²/d · 4⁻ᵇ.

### 4.4 Componente 3: Alocador Global de Bits (Contribuição Original)

**Formulação**: dado orçamento total de memória `B_total`, modelo com `L` camadas, e contexto `n`, encontrar:

- `b_w^(l)`: bits de pesos da camada `l`
- `b_k^(l)`: bits da key cache da camada `l`
- `b_v^(l)`: bits da value cache da camada `l`

que minimizem distorção total agregada sob restrição de bits.

**Função objetivo**:

```
min  Σ_l [α_w(l) · D_w(b_w^(l)) + α_k(l) · n · D_k(b_k^(l)) + α_v(l) · n · D_v(b_v^(l))]

s.t. Σ_l [N_w(l) · b_w^(l) + n · d · b_k^(l) + n · d · b_v^(l)] ≤ B_total
     b_w, b_k, b_v ≥ 1
```

onde:
- `D_w, D_k, D_v` são bounds de distorção (usar TurboQuant Theorem 1 para todos)
- `α(l)` são pesos de sensibilidade por camada (medidos via Eq. 3 do MLWQ — preservamos essa contribuição deles)
- `N_w(l)` é número de pesos da camada
- `n` é comprimento de contexto (entra para KV mas não para pesos — captura a dinâmica de scaling)

**Solução**: Lagrangiana dá condição de otimalidade do tipo:

```
∂D_x(b)/∂b · constante_x = λ  para todo x ∈ {w, k, v}, todo l
```

Como `D(b) ∝ 4⁻ᵇ`, a derivada é `-ln(4) · 4⁻ᵇ`. A solução é water-filling:

```
b_x^(l) = max(1, log₄(c_x^(l) / λ))
```

onde `λ` é ajustado por busca binária para satisfazer a restrição de orçamento. **Custo: O(L log(1/ε))**, trivial em comparação com a enumeração 2^l do MLWQ.

**Por que isso é a contribuição central**: nenhum dos métodos comparativos (AWQ, OPTQ, OWQ, SliM-LLM, MLWQ, OmniQuant, KIVI) tem alocação principled entre weights e KV cache. Todos usam bit-budgets fixos ou heurísticas locais. Esta é a primeira formulação que trata o problema como otimização global com solução em forma quase-fechada.

<!-- [ADIÇÃO V2.3 — variantes PPL-otimizada vs hardware-aligned] -->

#### 4.4.1 Duas variantes do alocador

A solução water-filling produz bit-widths reais (ex: 2.7, 3.4, 4.1) que não são diretamente executáveis em hardware. O arredondamento gera trade-off real entre qualidade e throughput, que vamos reportar explicitamente como **duas variantes do método**:

**Variante A — PPL-otimizada (`Ours-Free`)**:
- Bit-widths arredondados para inteiros arbitrários (qualquer valor ≥ 1)
- Permite combinações como {2, 3, 4, 5} convivendo na mesma camada
- Maximiza qualidade (PPL, zero-shot)
- Custo: kernels CUDA precisam fazer bit unpacking arbitrário, throughput sofre
- Compatível com AutoGPTQ-style mixed-precision

**Variante B — Hardware-aligned (`Ours-HW`)**:
- Bit-widths restritos a `{2, 4, 8}` (potências de 2, alinhadas a byte boundaries)
- Compatível com kernels modernos otimizados (Marlin, Machete, INT4 TensorCores do Jetson)
- Sacrifica algum PPL em troca de throughput significativamente melhor
- Implementação: water-filling com restrição adicional `b ∈ {2, 4, 8}`, resolvida via projeção sobre conjunto discreto

**Por que reportar ambas**:

1. Honestidade científica: o trade-off PPL vs throughput é real e o leitor merece ver
2. Diferentes hardwares preferem diferentes pontos: RTX 4090 com Marlin → HW-aligned ganha; GPU sem kernels otimizados → Free ganha
3. Em Jetson Orin Nano especificamente: HW-aligned deve dominar porque INT4 tem suporte nativo, INT3/INT5 não
4. Evita armadilha do MLWQ que reporta só uma configuração e fica mais lento que FP16 em hardware moderno

A variante reportada como "principal" no paper depende dos resultados empíricos da Fase 6 (Jetson). Se HW-aligned mantiver qualidade próxima de Free com throughput muito melhor em edge, ela vira o método principal e Free vira ablação.

<!-- [/ADIÇÃO V2.3] -->

### 4.5 Integração com Flash Attention

**Problema**: TurboQuant requer rotação aleatória de query no momento do attention; isso pode quebrar padrões de Flash Attention 2/3.

**Solução proposta**:

- Rotação compartilhada `Π` é aplicada uma vez ao tensor de query inteiro antes do FA
- Keys e values já chegam rotacionados ao serem armazenados na cache
- FA opera sobre vetores rotacionados sem modificação interna
- Custo adicional: 1 multiplicação Hadamard por forward pass

Especificar isso explicitamente no paper evita objeção comum de revisores.

<!-- [ADIÇÃO V3 — análise teórica via Yao + SLB adaptada] -->

### 4.6 Análise Teórica: Lower Bound para Mixed-Precision Weight + KV Quantization

A contribuição teórica original do paper é um **lower bound informacional** para o problema combinado de quantização de pesos e KV cache sob restrição de bits totais. Seguimos o template Yao + Shannon Lower Bound do TurboQuant Theorem 3, adaptado para o setting mixed-precision.

#### 4.6.1 Setup formal

Dado:
- Modelo com `L` camadas, cada uma com matriz de pesos `W_l ∈ R^(d×d)`
- Sequência de tokens com KV cache `K_l, V_l ∈ R^(n×d)` por camada
- Orçamento total `B_total` bits
- Quantizador randomizado `Q: (W, K, V) → {0,1}^B_total`

Buscamos cota inferior para distorção total:

```
D_total = Σ_l [α_w(l) · D_w(W_l, Q) + n · α_k(l) · D_k(K_l, Q) + n · α_v(l) · D_v(V_l, Q)]
```

#### 4.6.2 Estratégia de prova

**Passo 1 — Yao's Minimax**: O desempenho do melhor algoritmo randomizado sobre inputs adversariais é igual ao do melhor algoritmo determinístico sobre alguma distribuição. Escolhemos distribuição produto: pesos uniformes na esfera por canal, KV vetores uniformes na esfera unitária.

**Passo 2 — SLB por componente**: Para cada componente (pesos, K, V), aplicamos Lemma 3 do TurboQuant. Cada um contribui distorção mínima `≥ 4⁻ᵇ` onde `b` é o bit-width específico daquele componente.

**Passo 3 — Constraint Lagrangiana**: Sob restrição `Σ b · n_dims = B_total`, minimizar soma de bounds individuais é problema de otimização convexa com solução water-filling. A solução ótima é exatamente a alocação que nosso método propõe.

**Passo 4 — Pigeonhole**: Para extrair bound de inner product (necessário para attention), aplicamos pigeonhole sobre coordenadas, similar ao Theorem 3.

#### 4.6.3 Resultado esperado

O bound inferior tem forma:

```
D_total ≥ C(L, n) · 4^(-B_total / N_total)
```

onde `C(L, n)` é constante que depende da estrutura do modelo e contexto, e `N_total` é número total de coordenadas a quantizar. A demonstração de que **nosso método atinge esse bound dentro de fator constante** (a partir dos bounds do TurboQuant Theorem 1 e 2) seria a contribuição teórica central.

**Importante — limitação honesta**: o bound assume distribuição uniforme na esfera para pesos, o que **subestima o que é alcançável** quando se explora estrutura conhecida (sensibilidade inter-camada heterogênea). O bound é válido para algoritmos data-oblivious, e nossa postura híbrida (data-dependent leve para pesos) potencialmente bate o bound em prática — o que se traduz em vantagem empírica sobre métodos puramente data-oblivious como TurboQuant aplicado diretamente a pesos.

**Status no paper**: derivação completa do bound vai para apêndice. Resultado principal (statement do teorema) fica no corpo. Validação empírica (gap entre bound teórico e MSE empírico) entra como figura na Seção 6.

<!-- [/ADIÇÃO V3] -->

---

## 5. Plano Experimental

### 5.1 Modelos

**Prioridade alta**:
- SmolLM2-135M
- SmolLM2-360M
- Llama-3.2-1B

**Prioridade média**:
- Llama-3.2-3B
- Phi-2

**Critério**: SLM = ≤ 3B parâmetros. Acima disso sai do escopo declarado.

### 5.2 Hardware

- **Datacenter**: RTX 4090 24GB (replicar setup do MLWQ original)
- **Edge** (diferencial em relação ao MLWQ): Jetson Orin Nano 8GB

> **Justificativa**: o MLWQ se positions como solução para edge mas só avalia em RTX 4090. Validar em Jetson transforma o paper de "improvement on MLWQ" para "deployment-validated method".

### 5.3 Comprimentos de contexto

512, 1024, 2048, 4096, 8192, 16384

(32K se viável em Jetson com método proposto)

### 5.4 Métodos avaliados

| ID | Método | Pesos | KV cache |
|----|--------|-------|----------|
| M0 | FP16 baseline | FP16 | FP16 |
| M1 | MLWQ original (reprodução) | 3-bit MLWQ | FP16 |
| M2 | MLWQ + KV int4 (baseline trivial) | 3-bit MLWQ | int4 token-wise |
| M3 | MLWQ + KV int8 | 3-bit MLWQ | int8 token-wise |
| M4 | Nosso método (rotation Lloyd-Max) | 3-bit ours | FP16 |
| M5 | Nosso + QJL (ablação) | 3-bit ours | QJL puro |
| M6 | Nosso + TurboQuant | 3-bit ours | TurboQuant 3-bit |
| M7 | **Método completo** (alocador global) | adaptativo | adaptativo |
| M8 | KIVI (baseline KV state-of-the-art) | FP16 | KIVI 2-bit |
| M9 | PolarQuant (citação contextual) | FP16 | PolarQuant |

### 5.5 Métricas

**Qualidade**:
- WikiText2 PPL
- Zero-shot avg (PIQA, ARC-Easy, Winogrande, MathQA, LogiQA, ANLI_R2)
- LongBench-E (single-doc QA, multi-doc QA, summarization)
- Needle-in-Haystack score
<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
- **MMLU** (5-shot) — testa conhecimento estruturado e raciocínio; capacidades concentradas em poucos canais que rotação pode afetar mais que PPL ou commonsense reasoning
- **GSM8K** (8-shot) — raciocínio matemático multi-step; canário sensível para degradação de quantização que não aparece em PPL
- **ARC-Challenge** (separado de ARC-Easy) — testa raciocínio onde MLWQ original perde para OWQ; benchmark crítico para validar que a rotação não regride preservação de outliers
<!-- [/ADIÇÃO V2] -->

**Memória**:
- Weight memory (MB)
- KV cache memory (MB) — função do contexto
- Peak VRAM (MB)
- Compression ratio total
<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
- **Compression ratio efetivo (RM ratio)**: peak runtime memory FP16 / peak runtime memory quantizado, incluindo pesos + KV cache + ativações + buffers. **Distinção crítica em relação ao "weight memory ratio" (WM ratio) que MLWQ reporta**: no Llama-3.2-3B do paper original, WM ratio é ~6× (6.13G → 1.02G) mas RM ratio é apenas ~3.5× (6.17G → 1.78G). A diferença é evidência empírica direta da insuficiência weight-only — ativações e KV cache não quantizados consomem o ganho. Reportar ambos os ratios torna a vantagem do nosso método sobre MLWQ visível e quantitativa.
- **Memory footprint vs context length**: gráfico canônico do paper, mostrando que RM ratio do MLWQ degrada com contexto crescente enquanto o nosso método mantém ratio estável.
<!-- [/ADIÇÃO V2] -->

**Velocidade**:
- Prefill time (ms) por contexto
- Decode time per token (ms)
- Tokens/s
- Overhead de quantização/dequantização (ms)

**Robustez**:
- Variação por seed (3 seeds)
- Out-of-domain calibration (calibrar com WikiText2, avaliar em código/conversação)
<!-- [ADIÇÃO V3 — robustez e teoria] -->
- **Avaliação cross-dataset**: PPL em WikiText2 (calibration set) vs C4 (out-of-distribution) vs PTB. MLWQ original avalia apenas em WikiText2 (calibration = avaliação), o que superestima qualidade. Reportar gap WikiText2 → C4 quantifica generalização real do método.

**Métricas teóricas** (diferencial sobre MLWQ que não tem nenhuma):
- **Gap contra Shannon Lower Bound**: razão `MSE_empirico / 4⁻ᵇ` plotada em função de `b`. TurboQuant atinge fator ≈ 2.7. Nosso método herda valor próximo (com penalidade de Hadamard randomizada vs Haar exata, estimada 1.2-1.5×). MLWQ não reporta essa métrica e o valor dele é desconhecido — vamos calcular empiricamente para comparação.
- **Gap teórico do alocador Lagrangiano**: comparação entre `D_total` predito pelo bound teórico (Seção 4.6) e `D_total` empírico. Se o gap for pequeno (~constante), valida que o alocador está perto do ótimo informacional.
<!-- [/ADIÇÃO V3] -->

### 5.6 Critérios de sucesso (pré-registrados)

| Métrica | Limiar contra MLWQ |
|---------|---------------------|
| PPL piora | ≤ 3% (corrigido — antes era 5-10%, frouxo demais) |
| Zero-shot avg | queda ≤ 0.5 ponto |
| NIH score | ≥ 95% do FP16 |
| Peak VRAM (contexto 8K) | redução ≥ 30% |
| KV memory | redução ≥ 4× |
| Decode tokens/s | queda ≤ 10% (em RTX 4090) e ≤ 15% (em Jetson) |
| Prefill overhead | ≤ 25% |
<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
| MMLU 5-shot | queda ≤ 1.5 ponto contra MLWQ |
| GSM8K 8-shot | queda ≤ 2 pontos contra MLWQ (raciocínio matemático é mais frágil) |
| ARC-Challenge | queda ≤ 1 ponto contra MLWQ (canário para regressão de raciocínio com rotação) |
| RM ratio (contexto 8K) | ≥ 4× (vs ~3.5× do MLWQ na mesma escala) |
<!-- [/ADIÇÃO V2] -->

> **Honestidade científica**: se ganhar memória mas perder >15% de throughput em Jetson, isso é falha de deployment, não vitória. Reportar como tal.

<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
> **Risco específico de regressão em raciocínio**: a análise crítica da Seção 6 do MLWQ identificou que MLWQ frequentemente perde para OWQ em ARC-Easy, sugerindo que preservação de outliers em FP16 ajuda em tarefas de raciocínio específico. Como a rotação Hadamard destrói outliers por construção, o nosso método tem **risco real de regressão em MMLU/GSM8K/ARC-Challenge**. Se isso ocorrer, duas opções: (1) reportar honestamente como trade-off do método, (2) considerar versão híbrida onde camadas críticas para raciocínio mantêm tratamento outlier-aware separado. Decisão depende dos números empíricos da Fase 5.

<!-- [/ADIÇÃO V2] -->

---

## 6. Sequência de Execução

### Fase 1 — Reprodução de baseline (semana 1-2)

- Reproduzir MLWQ em SmolLM2-135M e Llama-3.2-1B
- Reproduzir PPL e zero-shot dentro de margem de 5% das tabelas originais
- Implementar baseline FP16 e KV int4 simples

**Gate**: se MLWQ não reproduz, usar GPTQ + AWQ como base alternativa e pivotar.

### Fase 2 — Demonstração da limitação do MLWQ (semana 3)

**Crítica**: esta fase decide o destino do paper. Se KV cache não dominar memória em nenhum modelo do escopo, a tese cai.

- Medir decomposição de memória (pesos vs KV) para todos os modelos × contextos
- Plotar gráfico: eixo X = contexto, eixo Y = memória total decomposta
- Identificar ponto de crossover
<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
- **Reportar separadamente WM ratio (compressão de pesos) e RM ratio (compressão runtime total incluindo ativações e KV cache)**. Evidência empírica esperada: WM ratio fica constante com contexto (pesos não escalam com n), mas RM ratio degrada conforme contexto cresce. A divergência entre WM e RM é o argumento empírico central do paper — quanto maior o gap, mais forte a tese de "weight-only é insuficiente".
<!-- [/ADIÇÃO V2] -->

**Entrega mínima desta fase**:
- Tabela 1: FP16 vs MLWQ em PPL, VRAM, tokens/s
- Tabela 2: decomposição de memória por contexto
- Gráfico 1: memória total vs contexto
- Gráfico 2: ponto a partir do qual KV cache > pesos quantizados
<!-- [ADIÇÃO V2 — análise crítica da Seção 6 do MLWQ] -->
- Gráfico 3: WM ratio vs RM ratio em função do contexto, demonstrando que o ratio efetivo do MLWQ degrada de ~6× para ~3.5× ou pior conforme o contexto cresce (replicando empiricamente a observação extraída da Tabela 7 do paper original)
<!-- [/ADIÇÃO V2] -->

**Gate de decisão**: se KV cache não passa pesos antes de 32K em nenhum modelo, considerar mudança de escopo.

### Fase 3 — Análise de tempo (semana 4)

Decomposição honesta de tempo por componente em RTX 4090 e Jetson:

- Tempo de matmul de attention
- Tempo de dequantização de pesos
- Tempo de dequantização de KV cache
- Tempo de rotação Hadamard
- Tempo de FFN

**Crítica importante**: sem isso, podemos otimizar memória e descobrir tarde que latência ficou pior em edge. Esta análise não estava no plano original e é essencial.

### Fase 4 — Implementação do método (semana 5-7)

- Componente 1: rotação Hadamard + Lloyd-Max para pesos
- Componente 2: TurboQuant_prod para KV cache
- Componente 3: alocador global Lagrangiano
- Integração com Flash Attention 2

### Fase 5 — Ablação (semana 8)

| ID | Configuração | Pergunta respondida |
|----|--------------|---------------------|
| A1 | Pesos uniformes, sem rotação | A rotação importa? |
| A2 | Pesos rotacionados + Lloyd-Max, KV FP16 | Quanto vem da quantização de pesos? |
| A3 | Pesos MLWQ original, KV TurboQuant | Quanto vem da KV? |
| A4 | Bits fixos (3w + 4kv) | Quanto vem do alocador? |
| A5 | Sem QJL no resíduo (só MSE) | QJL contribui? |
| A6 | Método completo | Soma de todos os ganhos |
| A7 | <!-- [ADIÇÃO V2] --> Replicação de Obs 1 e Obs 2 do MLWQ em ambiente controlado: histograma 1D de saliência por canal, em múltiplos modelos (SmolLM2-135M, Llama-3.2-1B, Phi-2) e múltiplos blocos (não só bloco 3), pré e pós-rotação | A motivação empírica do MLWQ generaliza? A rotação refuta a estrutura trimodal? |
| A8 | <!-- [ADIÇÃO V2.3] --> Lloyd-Max + rotação **sem TQP** vs com TQP grupo-específico vs com LWC global (1 grupo) | TQP ainda agrega após substituirmos quantizador uniforme por Lloyd-Max? Pode ser eliminado para simplificar o método? |
| A9 | <!-- [ADIÇÃO V2.3] --> Variante hardware-aligned `{2, 4, 8}` vs variante PPL-otimizada (bits arbitrários) | Qual o custo real em PPL de restringir bits a kernels otimizados? |
| A10 | <!-- [ADIÇÃO V3] --> Variante "two-stage radical" para pesos: Lloyd-Max com `b-1` bits + QJL no resíduo com 1 bit (análogo ao TurboQuant_prod aplicado a pesos) vs nosso método base | A construção two-stage do TurboQuant_prod, originalmente para KV cache, traz garantia de unbiased inner-product também para pesos? Vale a complexidade adicional? |
| A11 | <!-- [ADIÇÃO V3] --> Hadamard randomizada (custo `O(d log d)`) vs rotação Gaussian densa (custo `O(d²)`) | Qual o custo real em PPL de usar Hadamard em vez da rotação Haar exata? Trade-off com viabilidade em edge é favorável? |
| A12 | <!-- [ADIÇÃO V3] --> Lloyd-Max com codebook Gaussian asymptótico vs codebook Beta exato | Para SmolLM2-135M (d=576), aproximação Gaussian é boa o suficiente ou Beta exato muda os centróides materialmente? |

<!-- [ADIÇÃO V2 — leitura crítica da Seção 4 do MLWQ] -->

**Sobre A7 especificamente**: este experimento tem dupla função. Primeiro, fortalece a evidência empírica do nosso paper em pontos onde o MLWQ é fraco (amostra de 1 bloco, sem histograma 1D, sem cross-modelo). Segundo, se a hipótese de refutação da Observation 2 se confirmar, isso vira figura de destaque do paper: histograma pré-rotação trimodal vs pós-rotação Gaussian. **Tom diplomático na escrita**: não posicionar como "ataque ao MLWQ" — autores podem ser revisores. Frasear como "estendemos a análise empírica e observamos que a estrutura de saliência muda qualitativamente sob preconditioning rotacional, sugerindo que o tratamento trimodal pode ser substituído por quantização escalar sobre distribuição rotacionada".

<!-- [/ADIÇÃO V2] -->

<!-- [ADIÇÃO V2.3 — ablação A8 sobre eliminação do TQP] -->

**Sobre A8 especificamente — hipótese de simplificação radical**: a análise crítica da Seção 6 do MLWQ revelou que TQP é o componente que mais agrega (ablation §6.4 do paper original). Mas há leitura alternativa: TQP pode ser tão importante porque o quantizador uniforme da Eq. 1 do MLWQ é subótimo, e ajustar γ/β corrige parcialmente. Se nosso método usa Lloyd-Max sobre distribuição rotacionada (que é provadamente ótimo escalar para essa distribuição via TurboQuant Theorem 1), **TQP pode agregar pouco ou nada**. Se A8 confirmar isso, eliminamos TQP e ganhamos:
1. Argumento de elegância — método mais simples que MLWQ
2. Pipeline mais rápido (sem otimização gradiente de γ/β por grupo)
3. Defesa contra acusação de "incrementalismo" — não copiamos o componente mais inovador do MLWQ, substituímos a base inteira

Se A8 mostrar que TQP ainda agrega significativamente mesmo após Lloyd-Max+rotação, mantemos TQP no método (com adaptação para grupos pós-rotação, possivelmente n=1 ou n=2). Decisão é orientada pelo dado, não pela preferência teórica.

<!-- [/ADIÇÃO V2.3] -->

### Fase 6 — Validação em Jetson (semana 9)

- Portar implementação para Jetson Orin Nano
- Medir as mesmas métricas
- Comparar gap entre datacenter e edge

### Fase 7 — Escrita (semana 10-12)

---

## 7. Estrutura do Paper

1. **Introduction** — SLMs em edge, gap do MLWQ em long-context, contribuição
2. **Background** — weight quantization, KV cache, MLWQ, TurboQuant
3. **Motivation** — análise empírica da limitação do MLWQ (Fase 2)
4. **Method**
   - 4.1 Postura informacional híbrida (data-dependent leve + data-oblivious estrito)
   - 4.2 Random rotation Hadamard + Lloyd-Max para pesos
   - 4.3 TurboQuant_prod para KV cache
   - 4.4 Alocador global via water-filling Lagrangiano
   - 4.5 Integração com Flash Attention
5. **Theoretical Analysis** — bound de distorção total + lower bound adaptado via Yao + SLB
6. **Experiments**
   - 6.1 Setup
   - 6.2 Limitação do MLWQ (Fase 2): WM ratio vs RM ratio
   - 6.3 Quality preservation (PPL, zero-shot, MMLU, GSM8K, ARC-C, NIH, LongBench)
   - 6.4 Memory and throughput
   - 6.5 Gap teórico vs empírico (validação do bound)
   - 6.6 Edge deployment (Jetson)
   - 6.7 Ablation (A1-A12)
7. **Discussion** — quando o método não funciona, casos de borda
8. **Limitations** — kernels não otimizados ao máximo, modelos > 3B fora do escopo, prefill longo fora do escopo
9. **Conclusion**

---

## 8. Riscos e Planos B

| # | Risco | Probabilidade | Plano B |
|---|-------|---------------|---------|
| 1 | MLWQ difícil de reproduzir | Média | GPTQ/AWQ como base alternativa |
| 2 | KV int4 trivial já resolve | Alta | Foco no alocador global como contribuição central |
| 3 | TurboQuant lento em Jetson | Alta | Reportar como ganho de memória, não throughput; otimizar kernels Hadamard |
| 4 | KV cache não domina em SLM 135M | Alta | Subir escopo para Llama-3.2-3B, contexto 32K |
| 5 | Out-of-domain calibration falha | Média | Reportar como fraqueza, propor calibration mista |
| 6 | Rotação quebra Flash Attention | Média | Implementação cuidadosa documentada na seção 4.5 |
| 7 | Contribuição vista como combinação | Alta | Ênfase em formalização matemática do alocador + lower bound |
<!-- [ADIÇÃO V3] -->
| 8 | Hadamard randomizada não basta para preservar Lemma 1 em d pequeno | Média | Fallback para rotação Gaussian densa em modelos com d < 1024 (apenas SmolLM2-135M); reportar trade-off |
| 9 | Lower bound adaptado tem prova frouxa | Média | Apresentar bound mais simples (sem Yao adaptado) com gap maior contra TurboQuant; ainda é primeiro lower bound no setting |
| 10 | Canário (TurboQuant em SmolLM2) já resolve sem método novo | Alta | Pivotar paper para "alocador global + Jetson validation" como contribuição principal, com TurboQuant aplicado a SLM como baseline reportado |

---

## 9. Diferenciais que Tornam o Paper Defensável

1. **Formalização matemática do alocador global** (não heurística) — water-filling com solução em forma quase-fechada
2. **Validação em hardware edge real** (Jetson Orin Nano) — MLWQ não tem isso
3. **Lower bound informacional** adaptado para mixed-precision weight + KV — primeiro nesse setting
4. **Análise de crossover de memória** com contexto — quantitativa, não anedótica
5. **Out-of-domain calibration** — relevante para deployment real, ausente nos baselines
6. **Decomposição honesta de tempo** — evita armadilha de otimizar memória e perder latência
<!-- [ADIÇÃO V3] -->
7. **Postura informacional híbrida explícita** — data-dependent leve para pesos + data-oblivious para KV; nenhum trabalho prévio articula essa escolha
8. **Gap empírico contra Shannon Lower Bound reportado** — métrica teórica que MLWQ, AWQ, GPTQ, OmniQuant não reportam
9. **Hadamard randomizada com bounds adaptados** — escolha técnica explícita, não default; cita Ailon-Chazelle como base teórica
10. **Tratamento de norma fora do regime quantizado** com contabilidade honesta — evita armadilha de inflar compression ratio ignorando overhead

---

## 10. Próximos Passos Imediatos

<!-- [ADIÇÃO V3 — canário antes da Fase 1] -->

### 10.1 Canário pré-Fase 1 (semana 0)

**Pergunta crítica que decide o paper inteiro**: TurboQuant aplicado diretamente a SLM (sem nosso método) já resolve o problema?

**Procedimento** (3 dias de execução):

1. Setup ambiente: Llama-3.2-1B + SmolLM2-360M + Jetson Orin Nano
2. Rodar TurboQuant exatamente como publicado (sem modificações) em KV cache
3. Medir: PPL, NIH, LongBench, peak VRAM, tokens/s — em RTX 4090 e Jetson
4. Comparar com FP16 baseline e MLWQ baseline

**Critério de decisão**:

- **Se TurboQuant em SLM mantém qualidade dentro de 2% da FP16 e roda em Jetson com tokens/s aceitável**: o problema está essencialmente resolvido pelo TurboQuant aplicado a SLM. Pivotar paper para "alocador global pesos+KV + validação Jetson como contribuição principal", não tentar competir com TurboQuant em qualidade de KV cache.
- **Se TurboQuant em SLM perde mais que 5% de PPL ou não roda em Jetson**: há gap genuíno. Prosseguir com plano completo.
- **Se TurboQuant em SLM perde 2-5%**: zona cinzenta. Tomar decisão baseada em quanto da perda vem do método vs quanto vem de adaptação para SLM.

**Por que canário antes da Fase 1**: poupa potencialmente meses de execução baseada em premissa falsa. Custo do canário é 1 semana; custo de descobrir tarde é trimestre inteiro.

<!-- [/ADIÇÃO V3] -->

### 10.2 Fase 1 (após canário, se prosseguir)

1. **Setup ambiente**: Llama-3.2-1B + SmolLM2-135M + dependências (PyTorch, transformers, lm-eval)
2. **Implementar baseline FP16 + medições de memória/tempo decompostos**
3. **Reproduzir MLWQ** ou pivotar para GPTQ se necessário
4. **Executar Fase 2** — esta fase é o gate principal: se a limitação não aparecer empiricamente, repensar escopo antes de implementar método

> **Princípio guia**: a contribuição mais forte não é "combinamos MLWQ com TurboQuant". É **"weight-only quantization é insuficiente para SLM long-context, e o problema admite formulação como otimização global Lagrangiana com solução em forma quase-fechada, com lower bound informacional adaptado e validada em hardware edge real"**. Toda decisão técnica e experimental deve servir a essa narrativa.
