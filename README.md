# 🔥 FireWatch AI

### Sistema inteligente de detecção de fogo e fumaça utilizando Visão Computacional e Inteligência Artificial Generativa

O **FireWatch AI** é um projeto acadêmico desenvolvido para detectar possíveis ocorrências de fogo e fumaça em imagens, vídeos e transmissões de webcam, utilizando técnicas de Deep Learning e Visão Computacional.

O sistema utiliza o modelo **YOLO (You Only Look Once)** para identificar fogo e fumaça em tempo real, registrar as ocorrências e apresentar os resultados em uma interface interativa desenvolvida com Streamlit.

Além da detecção visual, o projeto incorpora Inteligência Artificial Generativa por meio da API da Groq, utilizando modelos da família GPT-OSS para produzir relatórios técnicos, resumos e respostas sobre os registros realizados pelo sistema.

> ⚠️ **Aviso:** O FireWatch AI é um protótipo acadêmico e experimental. Não substitui sistemas certificados de detecção, prevenção e alarme de incêndio, nem deve ser utilizado como único mecanismo de segurança.

---

## 📌 Sumário

1. [Objetivos do projeto](#-1-objetivos-do-projeto)
2. [Arquitetura geral](#-2-arquitetura-geral)
3. [Pipeline de processamento de imagens](#-3-pipeline-de-processamento-de-imagens)
4. [Inteligência Artificial e modelos utilizados](#-4-inteligência-artificial-e-modelos-utilizados)
5. [Datasets utilizados](#-5-datasets-utilizados)
6. [Treinamento do modelo](#-6-treinamento-do-modelo)
7. [Avaliação e métricas](#-7-avaliação-e-métricas)
8. [Integração com IA Generativa](#-8-integração-com-ia-generativa)
9. [Funcionalidades da aplicação](#-9-funcionalidades-da-aplicação)
10. [Instalação e execução](#-10-instalação-e-execução)
11. [Estrutura do projeto](#-11-estrutura-do-projeto)
12. [Testes](#-12-testes)
13. [Limitações e trabalhos futuros](#-13-limitações-e-trabalhos-futuros)
14. [Referências](#-14-referências)
15. [Integrantes](#-15-integrantes)

---

## 🎯 1. Objetivos do projeto

O principal objetivo do FireWatch AI é desenvolver uma aplicação capaz de identificar possíveis ocorrências de fogo e fumaça por meio de imagens e vídeos, utilizando modelos de Inteligência Artificial.

### Objetivos específicos

- Aplicar técnicas de Visão Computacional e Deep Learning.
- Treinar um modelo YOLO para identificar fogo e fumaça.
- Processar imagens, vídeos e transmissões de webcam.
- Implementar técnicas de pré-processamento de imagens.
- Registrar automaticamente as detecções realizadas.
- Apresentar alertas visuais e informações em um dashboard.
- Integrar Inteligência Artificial Generativa para produzir relatórios técnicos.
- Avaliar o desempenho do modelo por meio de métricas de classificação e detecção.

### Classes identificadas

| Classe | Identificador | Descrição |
|---|---:|---|
| Fire | 0 | Fogo |
| Smoke | 1 | Fumaça |

---

## 🏗️ 2. Arquitetura geral

O projeto combina duas abordagens de Inteligência Artificial que desempenham funções diferentes e complementares.

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| Visão Computacional | YOLO / Ultralytics | Identificação de fogo e fumaça |
| Processamento de imagens | OpenCV | Captura, leitura e transformação dos frames |
| Aplicação | Python / Streamlit | Interface e monitoramento |
| Armazenamento | CSV / arquivos locais | Histórico e registros das detecções |
| IA Generativa | GPT-OSS via Groq API | Interpretação dos registros e elaboração de relatórios |

### Fluxo geral do sistema

```mermaid
flowchart TD
    A[Imagem, vídeo ou webcam] --> B[Captura dos frames]
    B --> C[Pré-processamento com OpenCV]
    C --> D[Modelo YOLO treinado]
    D --> E{Existem detecções?}
    E -->|Sim| F[Identificação de fogo ou fumaça]
    E -->|Não| G[Continua o monitoramento]
    F --> H[Aplicação do threshold]
    H --> I[Alertas e imagens anotadas]
    I --> J[Registro em CSV e screenshots]
    J --> K[Dashboard e histórico]
    K --> L[Metadados e estatísticas]
    L --> M[IA Generativa via Groq]
    M --> N[Relatórios e respostas inteligentes]
```

A arquitetura separa a detecção visual da geração de informações em linguagem natural. Dessa forma, o modelo YOLO é responsável por identificar os objetos, enquanto o modelo generativo auxilia na interpretação e apresentação dos registros.

---

## 🖼️ 3. Pipeline de processamento de imagens

O pipeline representa todas as etapas realizadas desde a entrada de uma imagem ou vídeo até a apresentação e o armazenamento dos resultados.

### 3.1. Entrada de dados

O sistema aceita diferentes fontes de entrada:

- Imagens.
- Arquivos de vídeo.
- Transmissão de webcam.

No caso dos vídeos e da webcam, o OpenCV realiza a leitura e captura dos frames que serão encaminhados para o processamento.

### 3.2. Pré-processamento

Antes da inferência, o sistema pode aplicar transformações para preparar os frames.

As operações disponíveis incluem:

| Técnica | Objetivo |
|---|---|
| Resize | Redimensionamento da imagem |
| Gaussian Blur | Redução de ruídos e pequenas variações |
| Denoise | Redução de ruídos visuais |
| Ajuste de contraste | Alteração de brilho e contraste |
| BGR → RGB | Conversão de espaço de cores |
| BGR → HSV | Conversão para análise de matiz, saturação e valor |

O pré-processamento é configurável pela interface.

**Importante:** essas transformações são aplicadas na etapa de inferência, enquanto o treinamento utiliza as imagens originais preparadas pelo pipeline de datasets. O efeito de cada transformação precisa ser avaliado experimentalmente.

### 3.3. Detecção com YOLO

Após a preparação do frame, a imagem é encaminhada ao modelo YOLO treinado.

O modelo analisa os padrões visuais e retorna informações sobre os objetos identificados.

Entre os resultados estão:

- Classe identificada.
- Pontuação de confiança (*confidence*).
- Coordenadas da caixa delimitadora (*bounding box*).

As caixas delimitadoras permitem visualizar a localização dos possíveis focos de fogo e fumaça na imagem.

### 3.4. Aplicação do threshold

O threshold define a confiança mínima necessária para que uma detecção seja considerada válida.

O valor padrão utilizado na aplicação é **0.50 (50%)**, podendo ser alterado nas configurações.

Exemplo:

| Confiança identificada | Threshold | Resultado |
|---:|---:|---|
| 92% | 50% | Aceita |
| 68% | 50% | Aceita |
| 32% | 50% | Descartada |

Um threshold menor permite considerar mais detecções, mas pode aumentar os falsos positivos. Um threshold maior torna o sistema mais restritivo e pode aumentar os falsos negativos.

A confiança é uma pontuação produzida pelo modelo, não uma garantia de probabilidade real de incêndio.

### 3.5. Alertas e armazenamento

Quando uma detecção válida é identificada, o sistema pode:

- Exibir alertas visuais.
- Apresentar caixas delimitadoras na imagem.
- Registrar a ocorrência no histórico.
- Salvar capturas de tela.
- Atualizar os indicadores do dashboard.

Os registros são armazenados em arquivos CSV e as imagens correspondentes podem ser armazenadas na pasta de screenshots.

O sistema utiliza um intervalo mínimo entre registros (*cooldown*), configurado por padrão em 8 segundos por classe e origem, evitando a geração excessiva de registros repetidos.

### 3.6. Visualização dos resultados

As informações armazenadas alimentam a interface desenvolvida com Streamlit.

O usuário pode acompanhar o histórico, consultar indicadores, filtrar registros e exportar informações para análise.

---

## 🧠 4. Inteligência Artificial e modelos utilizados

O FireWatch AI utiliza modelos de IA especializados em duas tarefas: detecção visual e geração de linguagem natural.

### 4.1. YOLO — Visão Computacional

O YOLO (*You Only Look Once*) é uma arquitetura de Deep Learning utilizada para detecção de objetos.

No projeto, é utilizado o modelo **YOLO11n**, da família YOLO11, como peso inicial padrão para o treinamento.

A variante Nano foi escolhida por ser mais leve e adequada a experimentos acadêmicos que também podem ser executados em CPU.

O projeto possui fallback para YOLOv8n caso o peso padrão não esteja disponível. Também permite informar outro modelo por parâmetro.

Responsabilidades do YOLO:

- Identificar fogo.
- Identificar fumaça.
- Retornar a confiança das detecções.
- Localizar os objetos por meio de bounding boxes.
- Processar imagens e frames de vídeos e webcam.

O modelo treinado pelo projeto é armazenado em:

```text
models/fire_model.pt
```

### 4.2. GPT-OSS — Inteligência Artificial Generativa

Para a geração de relatórios e respostas inteligentes, o projeto utiliza modelos da família GPT-OSS, disponibilizados por meio da API da Groq.

| Modelo | Identificação | Utilização |
|---|---|---|
| GPT-OSS 120B | `openai/gpt-oss-120b` | Modelo padrão para relatórios e respostas |
| GPT-OSS 20B | `openai/gpt-oss-20b` | Alternativa mais leve |

O modelo utilizado pode ser configurado pela variável `FIREWATCH_GROQ_MODEL` ou pela interface, conforme a implementação disponível.

Responsabilidades da IA generativa:

- Elaborar relatórios técnicos.
- Resumir os registros das detecções.
- Apresentar estatísticas por classe.
- Identificar pontos de atenção.
- Responder perguntas sobre o histórico.

### 4.3. Por que utilizar dois modelos?

Os modelos possuem responsabilidades diferentes.

O YOLO realiza a análise visual e identifica possíveis ocorrências de fogo e fumaça.

Já o GPT-OSS recebe metadados e estatísticas produzidos pelo sistema e transforma essas informações em textos mais compreensíveis.

Essa integração permite combinar a capacidade de detecção visual com a geração automatizada de informações.

---

## 📚 5. Datasets utilizados

Para o treinamento e avaliação do modelo, o projeto utiliza datasets públicos de imagens anotadas de fogo e fumaça.

### 5.1. Indoor Fire Smoke Dataset

**Dataset principal do experimento padrão.**

- Aproximadamente 5.000 imagens reais disponíveis na fonte.
- Imagens de ambientes internos.
- Anotações de fogo e fumaça.
- Formato de bounding boxes.
- Divisão original de treino, validação e teste em 70/15/15.

Link: https://zenodo.org/doi/10.5281/zenodo.15826133

No modo acadêmico leve, o pipeline utiliza uma amostra reprodutível de 1.500 imagens, selecionadas com `seed=42`.

### 5.2. D-Fire Dataset

**Dataset adicional para experimentos e ampliação da diversidade visual.**

- Mais de 21 mil imagens.
- Imagens com fogo.
- Imagens com fumaça.
- Imagens contendo fogo e fumaça.
- Imagens sem ocorrência de fogo ou fumaça.
- Anotações no formato YOLO.

Composição informada pelo dataset original:

| Categoria | Quantidade de imagens |
|---|---:|
| Apenas fogo | 1.164 |
| Apenas fumaça | 5.867 |
| Fogo e fumaça | 4.658 |
| Sem fogo ou fumaça | 9.838 |

O dataset contém 14.692 bounding boxes de fogo e 11.865 bounding boxes de fumaça.

Links:
- Repositório: https://github.com/gaia-solutions-on-demand/DFireDataset
- Download e informações: https://github.com/gaia-solutions-on-demand/DFireDataset#download-links

### 5.3. Preparação dos datasets

O pipeline realiza automaticamente:

1. Download dos datasets selecionados.
2. Leitura das imagens e anotações.
3. Conversão das anotações para o formato YOLO.
4. Padronização dos identificadores das classes.
5. Remoção de duplicatas exatas utilizando hash SHA-256.
6. Divisão dos dados em treino, validação e teste.
7. Geração da configuração YAML utilizada pelo treinamento.

No caso do D-Fire, os identificadores originais das classes são remapeados para o padrão do projeto:

```text
0 = Fire
1 = Smoke
```

A deduplicação por SHA-256 identifica arquivos exatamente iguais, mas não elimina imagens visualmente semelhantes.

---

## ⚙️ 6. Treinamento do modelo

O treinamento utiliza transferência de aprendizado (*Transfer Learning*), partindo de pesos pré-treinados e realizando o ajuste do modelo para as classes de fogo e fumaça.

### 6.1. Configurações de treinamento

| Configuração | Modo acadêmico leve | Treinamento completo |
|---|---:|---:|
| Dataset | Indoor | D-Fire |
| Imagens | 1.500 | Conjunto disponível |
| Épocas | 10 | 30 |
| Resolução | 512 × 512 | 640 × 640 |
| Batch size | 4 | 4 |
| Seed | 42 | Configuração do pipeline |

Parâmetros adicionais:

| Parâmetro | Valor | Função |
|---|---:|---|
| Patience | 5 | Parada antecipada |
| Mosaic | 1.0 | Combinação de imagens |
| Mixup | 0.10 | Mistura de imagens |
| Scale | 0.50 | Variação de escala |
| Translate | 0.10 | Deslocamento |
| Degrees | 5.0 | Rotação |
| Fliplr | 0.50 | Espelhamento horizontal |

### 6.2. Saída do treinamento

Ao finalizar o treinamento, o melhor peso é copiado para:

```text
models/fire_model.pt
```

Esse arquivo é utilizado posteriormente pela aplicação para realizar as detecções.

Os arquivos de treinamento, gráficos, matriz de confusão e exemplos de predição são armazenados na pasta `runs/`.

---

## 📊 7. Avaliação e métricas

O desempenho do modelo é avaliado por meio de métricas utilizadas em tarefas de detecção de objetos.

### 7.1. Métricas utilizadas

| Métrica | Significado |
|---|---|
| Precision | Proporção de detecções positivas que estavam corretas |
| Recall | Proporção de objetos reais que foram identificados |
| mAP50 | Média da precisão considerando IoU de 0.50 |
| mAP50-95 | Média da precisão em diferentes limiares de IoU, de 0.50 a 0.95 |

Essas métricas permitem analisar a capacidade do modelo de identificar fogo e fumaça e avaliar a qualidade das detecções.

### 7.2. Experimento com diferentes thresholds

O projeto realiza experimentos utilizando cinco limiares de confiança:

```text
0.25
0.40
0.50
0.60
0.75
```

O objetivo é comparar como diferentes valores afetam o equilíbrio entre falsos positivos e falsos negativos.

De maneira geral:

- Threshold baixo: tende a aumentar o Recall, mas pode reduzir a Precision.
- Threshold alto: tende a aumentar a Precision, mas pode reduzir o Recall.

A interpretação depende dos resultados observados no conjunto de teste.

**Observação:** o mAP é normalmente calculado considerando diferentes limiares de confiança. Ao comparar execuções com thresholds fixos elevados, é importante interpretar as métricas com cuidado, especialmente o mAP.

### 7.3. Arquivos gerados na avaliação

```text
models/fire_model.pt
reports/evaluation_summary.csv
reports/data_runtime.yaml
runs/firewatch/train/
```

- `fire_model.pt`: pesos do modelo treinado.
- `evaluation_summary.csv`: métricas por threshold.
- `data_runtime.yaml`: configuração do dataset utilizado.
- `runs/firewatch/train/`: resultados e evidências do treinamento.

Os valores efetivos das métricas devem ser consultados nos arquivos gerados pelo experimento.

---

## 🤖 8. Integração com IA Generativa

A integração com Inteligência Artificial Generativa foi desenvolvida para ampliar as funcionalidades do sistema, permitindo transformar os registros técnicos das detecções em relatórios e respostas em linguagem natural.

A comunicação é realizada por meio da **Groq API**, utilizando modelos GPT-OSS.

### 8.1. Pipeline da integração

```mermaid
flowchart TD
    A[Detecções realizadas pelo YOLO] --> B[Histórico de registros]
    B --> C[Agregação de dados e estatísticas em Python]
    C --> D[Preparação dos metadados]
    D --> E[Envio para a Groq API]
    E --> F[Modelo GPT-OSS]
    F --> G[Resposta estruturada]
    G --> H[Validação do JSON Schema]
    H --> I[Relatório ou resposta ao usuário]
```

### 8.2. Relatórios inteligentes

O sistema permite gerar relatórios técnicos em português contendo:

- Resumo executivo.
- Estatísticas por classe.
- Análise temporal.
- Pontos de atenção.
- Recomendações de verificação humana.
- Conclusão.

Os relatórios podem ser exportados nos formatos TXT e PDF.

### 8.3. Assistente inteligente

O assistente permite realizar perguntas em linguagem natural sobre o histórico de detecções.

Exemplos:

- Quantas detecções de fogo foram registradas?
- Qual classe apareceu com maior frequência?
- Quais foram os principais registros?
- Faça um resumo das ocorrências.

O assistente utiliza até oito mensagens anteriores como contexto da conversa.

### 8.4. Resumo automático

Ao concluir o processamento de um vídeo, o sistema pode gerar automaticamente um relatório dos registros produzidos naquela execução.

A funcionalidade é ativada por padrão, mas pode ser desabilitada na interface para evitar chamadas desnecessárias à API.

### 8.5. Privacidade e confiabilidade

A integração foi desenvolvida com cuidados específicos:

- Apenas metadados e estatísticas são enviados à Groq.
- Imagens e vídeos não são enviados à API.
- Caminhos locais e coordenadas das bounding boxes não são transmitidos.
- Os cálculos estatísticos são realizados em Python.
- O modelo generativo é responsável pela redação e interpretação textual.
- As respostas dos relatórios são validadas por meio de JSON Schema.
- O sistema utiliza cache de sessão para reutilizar relatórios idênticos.

A IA generativa não realiza a detecção visual e não confirma a existência de um incêndio real.

Mais detalhes sobre a integração podem ser encontrados em:

`docs/IA_GENERATIVA_GROQ.md`

---

## 💻 9. Funcionalidades da aplicação

A interface foi desenvolvida utilizando Streamlit e está organizada em diferentes páginas.

| Página | Funcionalidade |
|---|---|
| Início | Indicadores, gráficos e visão geral |
| Monitoramento | Processamento de vídeos enviados pelo usuário |
| Webcam | Detecção em tempo real |
| Histórico | Consulta, filtros e exportação de registros |
| Relatórios Inteligentes | Geração de relatórios técnicos com IA |
| Assistente Inteligente | Perguntas em linguagem natural sobre o histórico |
| Configurações | Threshold, cooldown, pré-processamento e webcam |

---

## 🚀 10. Instalação e execução

### 10.1. Requisitos

- Python 3.10 ou superior.
- Conexão com a internet para instalação das dependências e download dos datasets.
- Windows recomendado para o script de instalação automática.
- GPU NVIDIA com CUDA opcional.

Sem uma GPU NVIDIA compatível, o pipeline pode ser executado em CPU, embora o treinamento e a inferência possam levar mais tempo.

### 10.2. Clonar o repositório

```bash
git clone https://github.com/vasconcelosguu/FireWatch.git

cd FireWatch
```

### 10.3. Criar ambiente virtual

```bash
python -m venv .venv
```

Ativar o ambiente no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Atualizar o pip:

```bash
python -m pip install --upgrade pip
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

### 10.4. Configurar a API da Groq

Para utilizar as funcionalidades generativas:

1. Criar uma chave de API na plataforma da Groq.
2. Copiar o arquivo `.env.example` para `.env`.
3. Preencher a variável `GROQ_API_KEY`.

Exemplo:

```env
GROQ_API_KEY=sua_chave_aqui
```

Não compartilhe sua chave e nunca publique o arquivo `.env` no GitHub.

### 10.5. Execução automática no Windows

O projeto possui um script que automatiza a instalação, preparação dos dados, treinamento e avaliação:

```powershell
.\setup_and_train.bat
```

Após a conclusão do pipeline, iniciar a aplicação:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

### 10.6. Execução manual do pipeline

Download do dataset Indoor:

```bash
python scripts/download_datasets.py --datasets indoor
```

Preparação dos dados:

```bash
python scripts/prepare_dataset.py --sources indoor --max-images 1500
```

Treinamento:

```bash
python scripts/train_model.py --epochs 10 --imgsz 512 --batch 4
```

Validação:

```bash
python scripts/validate_model.py
```

Experimentos com thresholds:

```bash
python scripts/evaluate_experiments.py
```

Também é possível executar o pipeline completo:

```bash
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10
```

Para utilizar os dois datasets:

```bash
python scripts/run_pipeline.py --sources dfire indoor --max-images 1500 --epochs 10
```

### 10.7. Teste individual do modelo

Imagem:

```bash
python scripts/test_model.py caminho/da/imagem.jpg
```

Vídeo:

```bash
python scripts/test_model.py caminho/do/video.mp4
```

Webcam:

```bash
python scripts/test_model.py 0
```

---

## 📁 11. Estrutura do projeto

```text
FireWatch/
│
├── app.py
├── requirements.txt
├── setup_and_train.bat
├── .env.example
├── .gitignore
│
├── .streamlit/
│   └── Configurações da interface
│
├── ai/
│   └── Prompts, relatórios e assistente inteligente
│
├── detector/
│   ├── detector.py
│   └── preprocess.py
│
├── interface/
│   ├── home.py
│   ├── monitor.py
│   ├── webcam.py
│   ├── history.py
│   ├── ai_reports.py
│   ├── ai_chat.py
│   └── settings.py
│
├── utils/
│   ├── video.py
│   ├── logger.py
│   └── alerts.py
│
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── evaluate_experiments.py
│   ├── run_pipeline.py
│   └── test_model.py
│
├── tests/
│   └── Testes automatizados
│
├── docs/
│   └── IA_GENERATIVA_GROQ.md
│
├── data_sources/
│   └── Datasets brutos (gerados)
│
├── datasets/
│   └── fire_smoke/
│       └── Dataset preparado
│
├── models/
│   └── fire_model.pt
│
├── reports/
│   ├── evaluation_summary.csv
│   └── data_runtime.yaml
│
├── runs/
│   └── Resultados do treinamento
│
├── screenshots/
│   └── Capturas das detecções
│
└── logs/
    └── Histórico das detecções
```

**Observação:** algumas pastas e arquivos são criados automaticamente durante o download dos datasets, treinamento e execução da aplicação. Os datasets brutos e os pesos treinados não precisam estar versionados no repositório devido ao tamanho dos arquivos.

---

## 🧪 12. Testes

O projeto possui testes automatizados para verificar o funcionamento das funcionalidades relacionadas à integração generativa.

Os testes contemplam:

- Preparação dos dados enviados à IA.
- Geração de relatórios estruturados.
- Funcionamento do assistente.
- Tratamento de erros.
- Validação das respostas.
- Remoção de caminhos locais e coordenadas dos dados enviados à API.

Para executar:

```bash
python -m unittest discover -s tests -v
```

Os testes da integração utilizam um cliente simulado e não precisam consumir chamadas reais da API da Groq.

---

## ⚠️ 13. Limitações e trabalhos futuros

### Limitações

- O desempenho depende da qualidade, resolução e iluminação das imagens.
- Ambientes externos e condições diferentes das presentes no dataset podem afetar os resultados.
- O dataset padrão é voltado a ambientes internos.
- A deduplicação atual remove arquivos idênticos, mas não necessariamente imagens visualmente semelhantes.
- O treinamento utiliza imagens originais, enquanto a inferência pode aplicar transformações de pré-processamento.
- Os registros do histórico respeitam um cooldown e, portanto, não representam necessariamente todas as detecções realizadas em cada frame.
- A geração de relatórios depende da disponibilidade e dos limites da API da Groq.
- O modelo generativo pode produzir interpretações incorretas, por isso os relatórios devem ser revisados.
- O protótipo não possui certificação para utilização em sistemas reais de segurança contra incêndio.

### Trabalhos futuros

- Ampliar a quantidade e diversidade dos dados de treinamento.
- Avaliar o desempenho em ambientes externos.
- Comparar diferentes variantes da arquitetura YOLO.
- Investigar o impacto individual das técnicas de pré-processamento.
- Melhorar a detecção em situações de baixa iluminação.
- Aprimorar a análise temporal das ocorrências.
- Evoluir os mecanismos de alerta e monitoramento.
- Expandir as funcionalidades do assistente inteligente.

---

## 📖 14. Referências

**Modelos e tecnologias**

- [Ultralytics — YOLO](https://docs.ultralytics.com/)
- [OpenCV — Documentação oficial](https://docs.opencv.org/)
- [Streamlit — Documentação oficial](https://docs.streamlit.io/)
- [Groq — Documentação da API](https://console.groq.com/docs)
- [OpenAI — GPT-OSS](https://openai.com/index/introducing-gpt-oss/)

**Datasets**

- [Indoor Fire Smoke Dataset — Zenodo](https://zenodo.org/doi/10.5281/zenodo.15826133)
- [D-Fire Dataset — GitHub](https://github.com/gaia-solutions-on-demand/DFireDataset)

**Referência acadêmica do D-Fire**

Venâncio, P. V. A. B.; Lisboa, A. C.; Barbosa, A. V. *An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices*. Neural Computing and Applications, 2022.

---

## 👥 15. Integrantes

Projeto desenvolvido por:

1. **Felipe Piovesan**
2. **Frederico Faben Brumati**
3. **Gustavo Vasconcelos Solomão**
4. **Ruan Gimenes**
5. **Vitor do Nascimento Zuchieri**
6. **José Roberto Miranda Neto**

---

<div align="center">

### 🔥 FireWatch AI

**Inteligência Artificial aplicada à detecção de fogo e fumaça.**

Projeto acadêmico desenvolvido para aplicação prática dos conceitos de Visão Computacional, Deep Learning e Inteligência Artificial Generativa.

[Repositório no GitHub](https://github.com/vasconcelosguu/FireWatch)

</div>
