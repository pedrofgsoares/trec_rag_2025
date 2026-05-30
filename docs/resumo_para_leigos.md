# O projecto TREC BioGEN 2025 em linguagem simples

*Para quem quer perceber o que se fez, sem precisar de saber programação ou linguística computacional. Sem fórmulas, sem código, sem acrónimos por explicar.*

---

## 1. O que é isto, em uma frase

Construímos um sistema que, dada uma resposta médica, encontra os artigos científicos que **apoiam** ou **contradizem** cada uma das suas frases. E descobrimos, no meio do caminho, que a forma como esta classe de sistemas é tradicionalmente avaliada tem um problema importante — e construímos uma forma melhor de o fazer.

---

## 2. O desafio concreto

O ponto de partida é uma competição internacional anual organizada pelo NIST (o instituto americano de padrões) chamada **TREC**. Em 2025 existia uma sub-competição específica para problemas biomédicos: a **BioGEN**. A nossa parte chamava-se **Task A**.

**O problema:** alguém faz uma pergunta médica (*"a aspirina previne enfartes?"*). Outro sistema gera uma resposta de cinco ou seis frases. A nossa tarefa: para cada frase dessa resposta, encontrar **até três artigos** da base de dados PubMed (a maior base de dados médica do mundo) que a apoiem, e até três que a contradigam.

O catálogo que temos de vasculhar tem **26,8 milhões de artigos científicos**. Para 40 perguntas, com 4-8 frases cada, multiplicado por 26,8 milhões de candidatos a artigo — todas as combinações possíveis. Isto, num computador portátil.

**As restrições do projecto:**
- Um único portátil pessoal, ligado em casa, sem cloud institucional.
- 4 GB de memória de placa gráfica (o que limita brutalmente que modelos de inteligência artificial podem correr).
- Orçamento total para chamadas a APIs pagas: cerca de 8 euros.

---

## 3. Como se faz isto na prática — a analogia da biblioteca

Imagine uma biblioteca com 26,8 milhões de livros. Receberam uma frase (*"a aspirina previne enfartes em pacientes com história familiar de doença cardíaca"*) e querem que indiquem os três livros que melhor a sustentam.

O nosso processo tem cinco etapas:

**1. Procura rápida — o índice de palavras.**
É o índice da biblioteca. Pega nas palavras-chave (*"aspirina", "enfarte", "história familiar"*) e devolve em segundos os 100 livros mais prováveis. Não percebe o sentido das frases — só sabe contar palavras. Mas é rapidíssimo. **Reduzimos de 26,8 milhões para 100.**

**2. Avaliação por especialista.**
Pegamos nesses 100 e passamo-los a um modelo de inteligência artificial treinado especificamente em textos médicos do PubMed. Esse modelo lê a pergunta e cada artigo lado a lado, e dá uma nota. É mais lento mas muito mais preciso. **Reduzimos de 100 para 30.**

**3. Decisão semântica.**
Outro modelo, agora especializado em decidir se uma frase apoia, contradiz, ou é neutra em relação a outra. Lê cada um dos 30 e decide.

**4. Filtro de negação.**
Texto clínico tem armadilhas. *"Não há evidência de a aspirina prevenir enfartes"* tem quase as mesmas palavras que *"Há evidência de a aspirina prevenir enfartes"* mas significa o oposto. Este filtro detecta negações antes de tomarmos a decisão final.

**5. Selecção.**
Dos sobreviventes, escolhemos os 3 melhores apoiantes e os 3 melhores contraditores.

Tudo isto, **para cada frase de cada resposta**. Para as 40 perguntas, são aproximadamente 200 frases. Para cada frase, todo este encadeamento corre do início ao fim. Demora cerca de 6 horas.

---

## 4. O primeiro choque — perdemos por uma goleada

Existe uma referência oficial dos organizadores chamada **baseline**. É um sistema simples a que todas as equipas são comparadas. Marca **44 pontos** numa escala de 0 a 100.

O nosso sistema marcou **5,5 pontos**.

Uma diferença de quase 40 pontos. Em qualquer competição, isto seria uma derrota humilhante: reconstrução completa, começar do zero.

Excepto que não.

---

## 5. A descoberta — a régua estava torta

O TREC não consegue ter humanos a ler 26,8 milhões de artigos. Por isso, antes da competição, os organizadores **pré-seleccionam um conjunto pequeno** de artigos para humanos especialistas avaliarem. A esse conjunto chama-se *pool*.

E aqui está o truque que mudou tudo: **a pool foi construída a partir das próprias escolhas dos sistemas participantes.** Cada equipa contribui com uma escolha para a pool. O sistema baseline dos organizadores está na pool.

A regra, simples e brutal: **só os artigos que entraram na pool é que foram lidos por humanos**. Tudo o resto, mesmo que correcto, conta automaticamente como "errado".

Isto significa duas coisas:

**(a)** O nosso sistema, que escolhia artigos *diferentes* dos da pool — não necessariamente piores, apenas diferentes — pagava o preço total. Dos 1124 artigos que retornávamos, só ~30 estavam na pool. Os outros ~1094 nunca foram lidos por ninguém — mas contavam contra nós como se estivessem errados.

**(b)** O baseline marcou 44 pontos *parcialmente porque ele próprio ajudou a definir a pool*. Estava a marcar o próprio teste.

Quando recalculámos os números numa pool honesta (que descrevo a seguir):
- O **baseline desce** de **44 → 16 pontos**.
- O **nosso sistema sobe** de **5,5 → 16,4 pontos**.

Ou seja: estamos empatados, não a perder de 40 pontos. **A régua estava torta.** Não era o sistema; era a medição.

Em ciência, isto chama-se um *artefacto metodológico*: quando o resultado depende mais da forma como mediste do que daquilo que mediste.

---

## 6. A solução — chamar um juiz artificial

A correcção que propomos: usar um **modelo de linguagem grande** (do tipo ChatGPT) como **juiz adicional**. Para cada artigo que o nosso pipeline retorna mas que não estava na pool original, perguntar ao modelo: *"este artigo apoia, contradiz, ou é neutro em relação a esta frase?"*. O modelo lê os dois textos e classifica.

Mas isto levanta uma pergunta imediata: porque é que havemos de confiar nas respostas do modelo?

A resposta — e este é o ponto técnico mais importante do projecto — é: **só se ele concordar com humanos**. Tínhamos 588 julgamentos humanos disponíveis. Fizemos o modelo julgar exactamente esses mesmos 588 casos, e medimos quantas vezes concordou com os humanos. Se concordasse em mais de 85% dos casos, considerávamos que era um juiz válido para casos *novos*. Se concordasse em menos, descartávamo-lo.

---

## 7. O momento em que aprendi mais — o caso da J-curve

O primeiro prompt (instrução) que demos ao modelo era directo: *"diz se o artigo apoia, contradiz ou é neutro. Só esse julgamento, em formato estruturado"*.

Resultado: o modelo falhou. Concordava em apenas 75% dos casos — abaixo dos 85% que tínhamos definido como critério. E falhava de forma estranha: classificava como *"Neutro"* coisas que humanos diziam claramente *"Apoia"*.

Em vez de mudar o sistema (a tentação óbvia), abrimos quatro casos concretos e lemos-os à mão. Um deles era este:

**Frase:** *"Baixar a pressão arterial abaixo de 120/70 mmHg pode causar problemas cardíacos."*

**Artigo do PubMed:** falava da *J-curve* — um fenómeno bem conhecido em cardiologia em que, abaixo de certos valores de pressão arterial diastólica, o risco cardiovascular começa a subir de novo. Falava da falta de estudos randomizados para alvos abaixo de 80 mmHg. As guidelines de referência recomendavam alvos <140/90, não 120/70.

**Veredicto humano:** *"Apoia."* (Faz sentido — o artigo descreve mecanismos que tornam plausível a frase.)

**Veredicto do modelo:** *"Neutro."* (Não está literalmente escrito que <120/70 causa problemas.)

Mostrámos isto ao perito clínico do projecto. A leitura dele foi: o modelo **tem o conhecimento médico** para fazer esta inferência — J-curve implica que baixa pressão pode prejudicar; a falta de estudos torna o *"pode causar"* defensável; as guidelines em 140/90 (não 120/70) significam que não há suporte profissional para ir abaixo de 120/70. O que faltava ao modelo era *espaço* para articular esta cadeia de raciocínio antes de decidir.

Mudámos o prompt para incluir: *"antes de decidir, escreve 2-3 frases de raciocínio"*. Ao mesmo modelo, no mesmo conjunto de 588 casos. Resultado: concordância subiu de **75% → 89%**. Passou o critério.

Esta foi a lição mais transferível do projecto inteiro: **antes de mudar o sistema, lê os casos.** A análise quantitativa apontava para uma direcção; a análise qualitativa (4 casos concretos lidos por um perito) apontava para outra. A segunda estava certa.

---

## 8. As experiências controladas — o que importa e o que não importa

Com um juiz validado, pudemos finalmente fazer ciência. Construímos uma "régua honesta" — uma pool aumentada com 4 758 julgamentos do modelo — e remedimos todas as variantes do pipeline contra essa régua.

Testámos **oito variantes**:

- Com e sem o reranker especializado em PubMed — neutro (não fez grande diferença).
- Com e sem o filtro de negação — importante para o caminho dos contradicts.
- Substituir o nosso modelo de decisão por outro especializado em texto médico — perdeu (domínio específico não é uma vitória garantida).
- Três formas diferentes de **expandir a pergunta original** com termos extra — falharam todas.

**O resultado negativo mais interessante:** expandir a query *piorou* o sistema. Tentámos três vezes, de três formas diferentes:

1. Uma técnica clássica estatística da literatura IR.
2. Uma versão dessa técnica filtrada por modelo de linguagem (literatura recente).
3. Reescrita inteira da pergunta por modelo de linguagem.

As três falharam.

A nossa explicação: quando as perguntas são já longas e específicas (uma frase médica completa já contém todos os termos-chave), expandir só introduz ruído. Esta é uma conclusão com peso no campo — **três tentativas independentes a falhar é uma triangulação científica**, não um acidente.

---

## 9. Mas como sabemos se o nosso juiz está certo? — o tribunal de três

Tudo o que descrevi depende de um juiz LLM. E se esse juiz tiver ele próprio um viés sistemático? Como sabemos?

Resposta: **chamar mais juízes, de famílias completamente diferentes.**

- **Juiz 1:** GPT-4o-mini (OpenAI, EUA).
- **Juiz 2:** Llama-3.3-70B (Meta, EUA).
- **Juiz 3:** Qwen2.5-72B (Alibaba, China).

Três modelos. Três organizações em dois continentes. Três conjuntos de dados de treino completamente independentes. Mesmo prompt. Mesmos 5 398 casos para julgar.

Medimos a concordância entre cada par usando uma métrica chamada **Krippendorff α** — essencialmente: quão de acordo estão os juízes, depois de descontar o acordo que se obteria por puro acaso.

| Par de juízes | Concordância (α) | Interpretação |
|---|---:|---|
| GPT-mini ↔ Llama | 0.12 | acordo fraco |
| GPT-mini ↔ Qwen | 0.20 | acordo fraco |
| **Llama ↔ Qwen** | **0.60** | **acordo substancial** |

Pensa um momento sobre isto: dois modelos completamente independentes (Meta nos EUA, Alibaba na China), com arquitecturas diferentes, com dados de treino completamente disjuntos, concordam significativamente entre si. O *único* juiz que diverge dos outros é o GPT-mini.

**Diagnóstico:** o GPT-mini era o juiz outlier. Era ele que estava a "ver" contradições onde os outros dois não viam.

Isto é o equivalente metodológico de um sistema judicial: **três jurados independentes corroboram entre si, e o veredicto final é apenas o que tem unanimidade**. Construímos uma pool com apenas as classificações em que pelo menos dois dos três juízes concordam. É uma régua mais curta — mas é uma régua justa.

---

## 10. Os resultados em quatro frases

1. A diferença aparente de 38 pontos entre nós e o baseline era **maioritariamente artefacto da forma como o TREC avalia** — cerca de 28 pontos. Apenas ~10 pontos eram diferença real, e em sentido inverso ao que aparentava.

2. Numa régua honesta, o nosso sistema **empata** com o baseline nos artigos *apoiantes* e **bate-o em mais do dobro** nos *contraditórios*.

3. Os melhores sistemas da competição (a equipa CLaC com 67,7 pontos) usam um Large Language Model no próprio pipeline. Nós deliberadamente não o fizemos — limitámo-nos a usar LLMs como juiz. Esta foi a escolha que nos limitou o tecto da pontuação nos apoiantes.

4. A **infraestrutura metodológica** que construímos (juiz validado, três jurados, intervalos de confiança, métricas de calibração) **é a primeira deste tipo na história da BioGEN**. Os próprios organizadores reconhecem no paper oficial que a análise de concordância "fica para o futuro".

---

## 11. O que isto custou — o número importante para o leitor leigo

- **Dinheiro em APIs:** 5,15 dólares (cerca de 4,90 euros). Total.
- **Tempo de computação:** ~22 horas de GPU acumuladas.
- **Hardware:** um portátil pessoal de 12 GB de RAM com uma placa gráfica Quadro T1000 de 4 GB.
- **Tempo humano:** seis meses, um único developer.

Quero sublinhar isto: ciência de Information Retrieval ao nível TREC era, há cinco anos, território de laboratórios universitários com clusters de GPUs. Hoje, com escolhas cuidadas e uso disciplinado de APIs comerciais, é viável fazê-lo num portátil pessoal. Não é só uma anedota de orçamento — **é um sinal de que a barreira de entrada à investigação em IR está a baixar**.

---

## 12. O que não conseguimos fazer — em ordem de importância

1. **Não bater os top dos apoiantes.** As equipas vencedoras (CLaC, GEHC-HTIC, dal) usam LLMs no próprio pipeline. A nossa escolha de manter o LLM só como juiz limita-nos estruturalmente. A próxima iteração natural corrige isto.

2. **Não comparar directamente contra outras equipas.** Nenhuma das 7 equipas BioGEN 2025 publicou o seu código. Não temos forma de re-correr o sistema deles na nossa régua honesta. Aliás, isto é um problema *para a track* mais do que para nós — se nós publicarmos, somos os primeiros.

3. **Não correr a variante de retrieval híbrido.** Combinar busca lexical com busca densa exigia ~24 h de pré-processamento que adiámos.

4. **Não fizemos fine-tuning do modelo de decisão.** Treiná-lo em bases de dados especializadas (SciFact, HealthVer, BioNLI) provavelmente dá +2-5 pontos. Adiado para uma fase futura.

5. **Revisão clínica limitada.** O nosso perito médico reviu 12 casos. Para um paper peer-reviewed, faríamos 50+ com dois peritos.

---

## 13. As cinco lições que ficam

1. **Antes de mudar o sistema, lê os casos.** A J-curve é o exemplo da vida.

2. **A medição é uma escolha científica, não um dado.** A derrota de 38 pontos desapareceu quando mudámos a régua. Em IR, isto é endémico: quem define a pool define os vencedores.

3. **Triangulação multi-juror é a forma certa de validar juízes-LLM.** Um juiz único é uma testemunha sem corroboração.

4. **Resultados negativos contam.** As três falhas independentes da expansão de queries são uma contribuição científica genuína — não menos por serem negativas, talvez mais.

5. **Investigação séria em IR é hoje viável num portátil.** Cinco euros e seis meses chegam para chegar a conclusões publicáveis. A barreira de entrada baixou; vale a pena aproveitá-la.

---

## 14. Em uma frase, o arco

Começou como *"construir um pipeline para uma competição TREC"*. Acabou como *"repensar como a competição TREC se mede"*. O que tornaria este trabalho relevante para além do exercício académico não é o que o pipeline faz — é o que descobrimos sobre a forma como a comunidade científica mede progresso em retrieval biomédico.

---

*E, em uma frase para a família, numa conversa de domingo:*

> *"Construí, no portátil, um sistema que ajuda médicos a verificar se os artigos científicos confirmam ou desmentem o que se diz numa resposta médica — e, no caminho, descobri que a forma oficial de avaliar estes sistemas tinha um viés que ninguém tinha quantificado."*
