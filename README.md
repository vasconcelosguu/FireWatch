# FireWatch AI

**Detecção de fogo e fumaça em vídeo e webcam com visão computacional (YOLO) e relatórios técnicos gerados por IA generativa.**

Projeto acadêmico de Deep Learning que implementa um pipeline reprodutível de Machine Learning, da coleta de datasets públicos à inferência em vídeo, com geração de alertas, registros e dashboard.

> **Aviso:** o FireWatch AI é um protótipo acadêmico e não substitui sistemas certificados de detecção e alarme de incêndio.

---

## Sumário

1. [Resumo](#1-resumo)
2. [Arquitetura de IA](#2-arquitetura-de-ia)
3. [Datasets](#3-datasets)
4. [Metodologia](#4-metodologia)
5. [Instalação e execução](#5-instalação-e-execução)
6. [Avaliação e resultados](#6-avaliação-e-resultados)
7. [IA generativa: relatórios inteligentes](#7-ia-generativa-relatórios-inteligentes)
8. [Estrutura do projeto](#8-estrutura-do-projeto)
9. [Testes](#9-testes)
10. [Limitações e trabalhos futuros](#10-limitações-e-trabalhos-futuros)
11. [Referências](#11-referências)
12. [Integrantes](#12-integrantes)

---

## 1. Resumo

O sistema detecta duas classes, `Fire` (0) e `Smoke` (1), em vídeos e imagens de webcam utilizando o **Ultralytics YOLO**. Quando há detecção, a interface exibe um alerta visual em tempo real; capturas de tela e registros em CSV são gerados com intervalo mínimo entre ocorrências (cooldown configurável, 8 s por classe e origem por padrão), e alimentam o dashboard em Streamlit.

O pipeline dispensa coleta e anotação manual de imagens: os scripts baixam datasets públicos, normalizam as anotações, removem duplicatas exatas, criam as divisões treino/validação/teste, treinam o modelo e avaliam o resultado.

```text
Datasets públicos → Coleta automática → Deduplicação → Normalização das anotações
→ Treino / Validação / Teste → Data augmentation → YOLO
→ Métricas (Precision, Recall, mAP50, mAP50-95) → Inferência em vídeo/webcam
→ Alerta + Screenshot + Log → Dashboard → Relatório por IA generativa
```

Por padrão, utiliza-se uma amostra reprodutível (`seed=42`) de **1.500 imagens reais**, com divisão aproximada de 70% treino, 15% validação e 15% teste. Isso permite executar o experimento completo em CPU, sem GPU dedicada.

Os datasets brutos e o arquivo `fire_model.pt` não acompanham o repositório por seu tamanho; ambos são gerados pelos scripts do projeto.

---

## 2. Arquitetura de IA

O sistema combina duas camadas com funções distintas:

| Camada | Tecnologia | Função |
|---|---|---|
| Visão computacional | YOLO (Ultralytics) | Detecta fogo e fumaça em cada frame e retorna classe, confiança e bounding box. |
| IA generativa | LLM via Groq | Interpreta os registros do YOLO e redige relatórios técnicos em português. Recebe apenas metadados e estatísticas, nunca imagens ou vídeos. |

A IA generativa não substitui a detecção visual, e seus relatórios não constituem confirmação de incêndio real.

---

## 3. Datasets

| | Indoor Fire Smoke (padrão) | D-Fire (opcional) |
|---|---|---|
| Imagens | 5.000 reais, ambientes internos (~200,5 MB) | Mais de 21 mil |
| Anotações | Bounding boxes de fogo e fumaça | Bounding boxes YOLO (14.692 de Fire e 11.865 de Smoke) |
| Divisão original | 70/15/15 | Treino/teste |
| Fonte | [Zenodo, DOI 10.5281/zenodo.15826133](https://doi.org/10.5281/zenodo.15826133) | [GitHub](https://github.com/gaia-solutions-on-demand/DFireDataset) |

- O **D-Fire** é baixado preferencialmente do Kaggle, com fallback para o espelho do Hugging Face (timeout ampliado, `hf_xet` e retomada de download). O caminho utilizado é salvo em `data_sources/dfire/.source_path`, evitando novos downloads.
- No D-Fire, o conjunto de teste oficial é preservado; apenas o conjunto de treino original é dividido em treino e validação (15%).
- O uso de mais de uma fonte aumenta a diversidade visual e reduz a dependência de um único domínio. O conjunto de teste permanece sempre separado, para medir a capacidade de generalização.
- Para usar todas as imagens do Indoor, em vez da amostra: `python scripts/prepare_dataset.py --sources indoor --max-images 0`.

---

## 4. Metodologia

### 4.1 Preparação dos dados

1. **Download automático** dos datasets públicos.
2. **Normalização das anotações** para o formato YOLO, aceitando labels `.txt` (YOLO) e `.xml` (Pascal VOC). No D-Fire, os índices de classe são remapeados (original: 0 = smoke, 1 = fire) para o padrão do projeto (0 = Fire, 1 = Smoke).
3. **Deduplicação** por hash SHA-256 do arquivo, que remove apenas duplicatas exatas, e não imagens visualmente parecidas.
4. **Divisão** em treino, validação e teste, com amostragem reprodutível (`seed=42`) e proporção 70/15/15 preservada quando `--max-images` é usado.

### 4.2 Treinamento

Fine-tuning de um modelo YOLO pré-treinado. O peso base padrão é o **YOLO11n** (versão *nano*, leve para CPU), com fallback para o `yolov8n`; outro peso pode ser informado com `--model`.

| Configuração | Modo acadêmico leve (padrão) | Treinamento completo |
|---|---|---|
| Dataset | Indoor, 1.500 imagens | D-Fire |
| Épocas | 10 | 30 |
| Resolução (`imgsz`) | 512 | 640 |
| Batch | 4 | 4 |

Parâmetros comuns, definidos em `scripts/train_model.py`:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `patience` | 5 | Parada antecipada se não houver melhora |
| `mosaic` | 1.0 | Combinação de 4 imagens por amostra |
| `mixup` | 0.10 | Mistura de pares de imagens |
| `scale` | 0.50 | Variação de escala |
| `translate` | 0.10 | Deslocamento |
| `degrees` | 5.0 | Rotação máxima |
| `fliplr` | 0.50 | Espelhamento horizontal |

### 4.3 Pré-processamento na inferência

O módulo `detector/preprocess.py` é aplicado **somente na inferência** (vídeo e webcam), de forma configurável e independente do modelo: resize (960 px por padrão), Gaussian Blur, denoise, ajuste de contraste (ativado por padrão com alpha 1,08 e beta 2) e conversões BGR → RGB e BGR → HSV. O treinamento usa as imagens originais do dataset, sem essas transformações.

O projeto não pressupõe que cada transformação melhore o desempenho. A interface permite comparar a imagem original com a processada, e o efeito sobre as métricas deve ser verificado experimentalmente.

### 4.4 Experimento de thresholds

O modelo é avaliado no conjunto de teste com limiares de confiança `0.25`, `0.40`, `0.50`, `0.60` e `0.75`, registrando Precision, Recall, mAP50 e mAP50-95. O objetivo é evidenciar o compromisso entre falsos positivos e falsos negativos.

Observações para a interpretação:

- Por padrão, `validate_model.py` e `evaluate_experiments.py` avaliam com `imgsz=640`, enquanto o modo leve é treinado com `imgsz=512`.
- O mAP costuma ser calculado com um limiar de confiança baixo; ao aplicar limiares altos (como 0.75), os valores de mAP tendem a diminuir artificialmente. A comparação entre thresholds deve considerar principalmente Precision e Recall.

---

## 5. Instalação e execução

**Requisitos:** Python 3.10 ou superior (exigido pelo SDK `groq`; o `setup_and_train.bat` recomenda Python 3.11 ou 3.12) e conexão com a internet para o download dos datasets e dependências.

### 5.1 Execução automática (Windows)

```powershell
.\setup_and_train.bat
```

O script cria a `.venv`, instala as dependências, baixa e prepara o dataset, treina o modelo, copia o melhor peso para `models/fire_model.pt`, valida no conjunto de teste e executa o experimento de thresholds. Em seguida, inicie a aplicação:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

O script detecta GPU NVIDIA (CUDA) por meio do `nvidia-smi`. Sem ela, inclusive com GPU AMD no Windows, o treinamento é feito em CPU.

### 5.2 Execução manual

```powershell
# Ambiente
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Pipeline completo
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10

# Pipeline com os dois datasets
python scripts/run_pipeline.py --sources dfire indoor --max-images 1500 --epochs 10
```

Etapa por etapa:

```powershell
python scripts/download_datasets.py --datasets indoor
python scripts/prepare_dataset.py --sources indoor --max-images 1500
python scripts/train_model.py --epochs 10 --imgsz 512 --batch 4      # adicione --device 0 para GPU NVIDIA
python scripts/validate_model.py
python scripts/evaluate_experiments.py
```

Treinamento completo com o D-Fire:

```powershell
python scripts/download_datasets.py --datasets dfire
python scripts/prepare_dataset.py --sources dfire
python scripts/train_model.py --epochs 30 --imgsz 640 --batch 4 --device cpu
python scripts/validate_model.py
python scripts/evaluate_experiments.py
```

Teste rápido do modelo em imagem, vídeo ou webcam, fora da interface:

```powershell
python scripts/test_model.py caminho/da/imagem.jpg     # imagem
python scripts/test_model.py caminho/do/video.mp4      # vídeo (tecla Q para sair)
python scripts/test_model.py 0                         # webcam
```

### 5.3 Uso da aplicação

| Página | Função |
|---|---|
| Início | Visão geral, indicadores e gráficos do histórico |
| Monitoramento | Upload de vídeo (MP4, AVI, MOV) e processamento frame a frame |
| Webcam | Detecção em tempo real pela câmera local |
| Histórico | Filtros, tabela de registros e exportação em CSV |
| Relatórios Inteligentes | Relatório técnico por IA generativa (TXT e PDF) |
| Assistente Inteligente | Perguntas em linguagem natural sobre o histórico |
| Configurações | Threshold, cooldowns, pré-processamento, webcam e recarga do modelo |

O threshold de confiança padrão é 0,50. Alertas visuais aparecem a cada frame com detecção; o alerta sonoro (`winsound`) e a captura da webcam via DirectShow (`CAP_DSHOW`) são específicos do Windows.

---

## 6. Avaliação e resultados

Ao final do pipeline, são gerados:

```text
models/fire_model.pt              Melhor peso treinado
reports/evaluation_summary.csv    Métricas por threshold
reports/data_runtime.yaml         Configuração do dataset utilizado
runs/firewatch/train/             Curvas de treino, matriz de confusão e exemplos
```

As métricas reportadas são Precision, Recall, mAP50 e mAP50-95, calculadas no conjunto de teste. A matriz de confusão e as curvas são geradas automaticamente pelo YOLO.

<!-- Inserir aqui a tabela com os resultados obtidos (Precision, Recall, mAP50, mAP50-95)
     e imagens de exemplo de detecção (acertos, falsos positivos e falsos negativos). -->

---

## 7. IA generativa: relatórios inteligentes

O modelo padrão é o `openai/gpt-oss-120b`, acessado pela API da Groq e configurável por meio da variável `FIREWATCH_GROQ_MODEL` ou pela interface (alternativa mais leve: `openai/gpt-oss-20b`).

**Funcionalidades**

- **Relatórios Inteligentes:** relatório técnico em português com resumo executivo, estatísticas por classe, análise temporal, pontos de atenção, recomendações de verificação humana e conclusão, exportável em TXT e PDF. Relatórios idênticos (mesmos dados, modelo e versão do prompt) são reutilizados do cache da sessão, sem nova chamada à API.
- **Assistente Inteligente:** perguntas em linguagem natural sobre o histórico de detecções, com até oito mensagens anteriores como contexto.
- **Resumo automático:** na página de Monitoramento, ao concluir o processamento de um vídeo, é gerado um relatório apenas dos registros criados naquela execução. A opção vem ativada por padrão e realiza uma chamada à API por vídeo com novos registros; desmarque-a para não consumir tokens. Se não houver detecções ou a API estiver indisponível, o vídeo permanece processado e o histórico é preservado.

**Privacidade e confiabilidade**

- Somente metadados e estatísticas são enviados à Groq; imagens, vídeos, caminhos locais e coordenadas nunca são transmitidos.
- Os números dos relatórios são calculados em Python, e o modelo de linguagem apenas redige a interpretação.
- A resposta do relatório é exigida em JSON Schema estrito e validada antes de ser exibida.

**Configuração**

1. Copie `.env.example` para `.env`.
2. Defina `GROQ_API_KEY` com a sua chave.
3. Não publique o arquivo `.env` no repositório.

Detalhes sobre arquitetura, prompt, segurança, testes e limitações em [docs/IA_GENERATIVA_GROQ.md](docs/IA_GENERATIVA_GROQ.md).

---

## 8. Estrutura do projeto

```text
firewatch-ai/
├── app.py                      Aplicação Streamlit
├── setup_and_train.bat         Setup e treinamento automáticos (Windows)
├── requirements.txt
├── .env.example                Modelo de configuração da chave Groq
├── .streamlit/                 Configuração do Streamlit
├── ai/                         Prompts, relatórios e assistente (chat_assistant.py)
├── detector/                   Detecção YOLO e pré-processamento
├── interface/                  Páginas: home, monitor, webcam, history, ai_reports, ai_chat, settings
├── utils/                      Vídeo, logger e alertas
├── scripts/                    Download, preparo, treino, validação, experimentos e testes do modelo
├── tests/                      Testes unitários (relatórios e assistente)
├── docs/                       Documentação complementar
├── data_sources/               Datasets brutos (gerado)
├── datasets/fire_smoke/        Dataset normalizado (gerado)
├── models/                     Pesos treinados (gerado)
├── runs/ e reports/            Saídas de treino e avaliação (gerado)
└── screenshots/ e logs/        Evidências e histórico de detecções (gerado)
```

O script `scripts/create_synthetic_dataset.py` é uma ferramenta didática e experimental, e não faz parte do pipeline principal.

---

## 9. Testes

Os testes unitários cobrem a preparação dos dados enviados à IA, a geração estruturada de relatórios, o assistente, o tratamento de erros e a remoção de caminhos e coordenadas do payload. Utilizam um cliente simulado e não consomem a API da Groq.

```powershell
python -m unittest discover -s tests -v
```

---

## 10. Limitações e trabalhos futuros

**Limitações**

- O desempenho depende da qualidade e resolução das imagens, do domínio das câmeras, da iluminação, da distância, do tamanho do fogo, da densidade da fumaça e do ambiente (interno ou externo). O dataset padrão contém apenas ambientes internos. Um bom resultado no conjunto de teste não garante segurança em ambiente real.
- O treinamento usa imagens sem pré-processamento, enquanto a interface aplica, por padrão, resize e ajuste de contraste na inferência, o que pode gerar pequena diferença entre as condições de treino e de uso.
- A deduplicação remove apenas arquivos idênticos; imagens muito semelhantes podem aparecer em conjuntos distintos.
- Os registros do histórico respeitam um cooldown entre capturas, portanto não representam todas as detecções frame a frame.
- O alerta sonoro e a abertura da webcam são específicos do Windows.

**Trabalhos futuros**

- Treinar com o conjunto completo e combinar múltiplas fontes de dados.
- Avaliar o efeito de cada etapa de pré-processamento sobre as métricas.
- Testar o modelo em vídeos externos, de domínio distinto do treinamento.
- Comparar diferentes variantes do YOLO.

---

## 11. Referências

- Venâncio, P. V. A. B.; Lisboa, A. C.; Barbosa, A. V. *An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices*. Neural Computing and Applications, 2022. Dataset D-Fire: [github.com/gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset).
- Putra, A. K. *Indoor Fire Smoke Dataset*. Zenodo, 2025. DOI [10.5281/zenodo.15826133](https://doi.org/10.5281/zenodo.15826133).

---

## 12. Integrantes

- Felipe Piovesan
- Frederico Faben Brumati
- Gustavo Vasconcelos Solomão
- Ruan Gimenes
- Vitor do Nascimento Zuchieri
- José Roberto Miranda Neto
