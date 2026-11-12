# FireWatch AI — Integração de IA Generativa com Groq

## 1. Objetivo

A integração acrescenta uma camada de **IA generativa baseada em LLM** ao FireWatch AI. O YOLO continua responsável pela visão computacional: identifica regiões classificadas como `Fire` ou `Smoke` e registra classe, confiança, horário, origem e evidência no histórico. A IA generativa recebe um resumo estruturado desses registros e redige um relatório técnico em português brasileiro.

A IA generativa não substitui o YOLO, não executa uma segunda detecção visual e não confirma a existência de incêndio. Sua função é interpretar os metadados já produzidos e organizar uma análise legível para apoiar a conferência humana.

## 2. Arquitetura e fluxo de dados

```text
Vídeo / Webcam
      │
      ▼
YOLO — detecção de Fire e Smoke
      │  classe, confidence, bounding box
      ▼
DetectionLogger — logs/detections.csv
      │
      ▼
Página Relatórios Inteligentes (Streamlit)
      │  seleção explícita de grupos
      ▼
Preparação local dos dados (Python)
      │  contagens, média/mínima/máxima,
      │  período, origens e grupos de evidência
      ▼
Prompt técnico + JSON Schema estrito
      │
      ▼
Groq API — GPT-OSS 120B (padrão)
      │  resposta JSON estruturada
      ▼
Validação da resposta no Python
      │
      ▼
Montagem do relatório
      ├── Exibição no Streamlit
      ├── Download TXT
      └── Download PDF
```

### Arquivos principais

| Arquivo | Responsabilidade |
|---|---|
| `ai/prompts.py` | Prompt de sistema, instruções de análise e JSON Schema de saída. |
| `ai/report_generator.py` | Limpeza dos dados, métricas, cliente Groq, validação, montagem Markdown e exportação PDF. |
| `interface/ai_reports.py` | Seleção de registros, configuração do modelo, verificação de conexão e interface de geração/download. |
| `app.py` | Registra a página Relatórios Inteligentes na navegação. |
| `.env.example` | Exemplo seguro de configuração da chave e modelo. |
| `tests/test_ai_reports.py` | Testes unitários de dados, geração estruturada, erros e conteúdo do relatório. |

## 3. O que a IA recebe

A aplicação calcula localmente e envia à Groq apenas um JSON com:

- identificador do relatório;
- quantidade de detecções e de grupos de evidência;
- início e fim do período observado;
- origens registradas (vídeo, webcam ou imagem, quando disponível);
- quantidade por classe (`Fire` e `Smoke`);
- média, mínima e máxima das pontuações de confiança por classe e no conjunto;
- amostra cronológica dos grupos, com classes, quantidades e estatísticas correspondentes;
- limitações conhecidas dos dados.

As estatísticas agregadas consideram todos os registros selecionados. Quando existem mais de 60 grupos, os detalhes enviados individualmente à IA são amostrados de forma distribuída ao longo do período, para não privilegiar apenas os registros mais antigos.

**Não são enviados:** imagens, vídeos, coordenadas das caixas delimitadoras, caminhos locais, nomes de arquivos ou dados pessoais. O caminho de evidência pode ser usado internamente para agrupar linhas do histórico, mas é removido antes da criação do payload.

## 4. O que a IA generativa produz

A resposta é estruturada e depois formatada pelo FireWatch com as seguintes seções:

1. Resumo executivo.
2. Resultado quantitativo por classe, calculado em Python.
3. Análise técnica.
4. Análise temporal.
5. Pontos de atenção.
6. Recomendações de verificação humana.
7. Limitações e qualidade dos dados.
8. Conclusão cautelosa.

O relatório inclui período, origens, quantidade de registros, número de grupos, estatísticas por classe e uma tabela dos grupos de evidência com horários, origens, classes e pontuação média. A IA produz a interpretação em linguagem natural; o programa calcula os números para evitar que o modelo altere contagens e percentuais.

## 5. Prompt e controle da resposta

O prompt em `ai/prompts.py` define o comportamento de um analista técnico do FireWatch. Ele obriga o modelo a:

- escrever em português brasileiro, com linguagem objetiva e técnica;
- diferenciar Fire de Smoke;
- não interpretar confidence como probabilidade de incêndio ou como gravidade;
- não inferir causalidade entre fogo e fumaça;
- não inventar local, causa, duração, danos, vítimas ou responsáveis;
- não alegar que analisou imagens ou vídeos;
- reconhecer que agrupamentos do CSV não equivalem necessariamente a incidentes;
- recomendar validação visual e aplicação do protocolo local por pessoas responsáveis.

A API usa **Structured Outputs com JSON Schema estrito**. O código valida novamente a presença e os tipos das seções recebidas antes de permitir a exibição e exportação. Uma resposta vazia, malformada ou incompleta é rejeitada.

## 6. Configuração da API

1. Use Python 3.10 ou superior.
2. Crie uma chave em [Groq Console](https://console.groq.com/keys).
3. Na raiz do projeto, copie `.env.example` para `.env`.
4. Insira a chave no `.env`:

```dotenv
GROQ_API_KEY=sua_chave_real
FIREWATCH_GROQ_MODEL=openai/gpt-oss-120b
```

5. Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

6. Inicie o Streamlit:

```powershell
python -m streamlit run app.py
```

7. Abra **🧠 Relatórios Inteligentes**, selecione grupos, escolha o modelo, use **Testar conexão** e depois **Gerar relatório técnico com IA**.

O botão de teste consulta a lista de modelos e não solicita geração de texto. A geração só acontece após clique explícito no botão de relatório. Relatórios idênticos, com os mesmos dados, modelo e versão do prompt, são reutilizados no cache da sessão.

### Modelo

O SDK oficial está fixado em `groq>=1.7.0,<2` (versão publicada no PyPI em agosto de 2026, na consulta de 9 de novembro de 2026). O padrão é `openai/gpt-oss-120b`, escolhido para priorizar capacidade de análise. A interface também disponibiliza `openai/gpt-oss-20b` como alternativa mais leve. A disponibilidade final depende da conta e das condições vigentes da Groq. O ID pode ser alterado pela interface ou pela variável `FIREWATCH_GROQ_MODEL`.

Documentação oficial consultada:

- [Modelos disponíveis na Groq](https://console.groq.com/docs/models)
- [GPT-OSS 120B](https://console.groq.com/docs/model/openai/gpt-oss-120b)
- [Structured Outputs](https://console.groq.com/docs/structured-outputs)
- [Text Generation / Chat Completions](https://console.groq.com/docs/text-chat)
- [SDK oficial Groq para Python](https://github.com/groq/groq-python)

## 7. Diferença entre YOLO e LLM

| Aspecto | YOLO | LLM via Groq |
|---|---|---|
| Tipo de IA | Visão computacional / Deep Learning para detecção de objetos. | Modelo de linguagem generativo. |
| Entrada | Frames de vídeo ou imagens capturadas pela aplicação. | JSON com metadados e estatísticas do histórico. |
| Saída | Classes, pontuações de confiança e caixas delimitadoras. | Texto técnico estruturado em seções de relatório. |
| Papel no sistema | Detectar regiões visualmente semelhantes às classes treinadas. | Explicar os registros, apontar limites e recomendar verificações. |
| Pode confirmar incêndio real? | Não, por si só. | Não; nesta integração nem recebe as imagens. |

## 8. Benefícios e limitações

### Benefícios

- transforma registros técnicos em relatório legível;
- separa fatos quantitativos calculados pelo programa da interpretação em linguagem natural;
- padroniza a estrutura de relatórios;
- permite exportação TXT e PDF;
- reduz chamadas repetidas com cache de sessão;
- mantém a chave fora do código-fonte e do Git.

### Limitações

- a qualidade da análise depende da qualidade dos registros e do modelo selecionado;
- a IA não enxerga o vídeo nem as imagens nesta versão;
- não existe localização geográfica no CSV atual;
- o score de confiança do YOLO não é probabilidade calibrada;
- grupos do histórico não equivalem automaticamente a incidentes;
- resposta generativa pode conter interpretação imperfeita, mesmo com prompt e esquema estruturado; a conferência humana permanece obrigatória;
- conexão, disponibilidade de modelo, limites de uso e custos dependem da conta Groq.

## 9. Testes

Execute:

```powershell
python -m unittest discover -s tests -v
```

Os testes unitários cobrem:

- apenas Fire;
- apenas Smoke;
- Fire e Smoke juntos;
- vários registros no mesmo grupo;
- histórico sem detecções;
- campos incompletos;
- remoção de caminhos, coordenadas e textos desconhecidos do payload;
- uso do modelo e JSON Schema estrito;
- falha de API;
- resposta JSON inválida ou incompleta;
- conteúdo quantitativo e aviso de segurança do relatório.

Os testes unitários utilizam um cliente simulado e não consomem a API. O teste real de chave/modelo está disponível no botão **Testar conexão** dentro do aplicativo e deve ser executado depois que a chave for configurada.

## 10. Segurança da chave

- Nunca coloque a chave no código Python ou no README.
- Não envie `.env` para o GitHub; o `.gitignore` já o exclui.
- Compartilhe apenas `.env.example`, sem uma chave funcional.
- Se uma chave for publicada acidentalmente, revogue-a no console da Groq e gere outra.

## Assistente conversacional

A página **Assistente Inteligente** usa a mesma chave e o mesmo modelo Groq configurados no `.env`. Cada pergunta é respondida com base em estatísticas agregadas do histórico e em até 30 grupos de evidência amostrados cronologicamente. A conversa mantém até oito mensagens anteriores como contexto. O usuário pode limpar a conversa a qualquer momento.

O assistente não recebe mídia, caminhos de arquivo, coordenadas ou caixas delimitadoras. Ele não confirma incêndios e deve declarar quando o histórico não contém a informação solicitada.

## Resumo automático após vídeo

Na página de monitoramento, a opção **Gerar resumo automático com IA ao terminar o vídeo** vem ativada. Quando o processamento termina, a aplicação compara a quantidade de linhas no CSV antes e depois da execução e envia à função de relatório somente as novas detecções daquele vídeo. Se houver registros, a mesma geração estruturada do relatório técnico cria o resumo e a interface oferece TXT e PDF. A chamada só ocorre ao concluir um processamento solicitado pelo usuário; pode ser desativada no checkbox para economizar tokens. Falhas na Groq não interrompem o processamento nem removem registros.
