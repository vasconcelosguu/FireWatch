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
9. [Limitações e trabalhos futuros](#9-limitações-e-trabalhos-futuros)
10. [Referências](#10-referências)
11. [Integrantes](#11-integrantes)

---

## 1. Resumo

O sistema detecta duas classes, `Fire` (0) e `Smoke` (1), em vídeos e imagens de webcam utilizando o **Ultralytics YOLO**. Cada detecção gera alerta, captura de tela e registro em log, que alimentam um dashboard em Streamlit.

O pipeline dispensa coleta e anotação manual de imagens: os scripts baixam datasets públicos, normalizam as anotações, removem duplicatas, criam as divisões treino/validação/teste, treinam o modelo e avaliam o resultado.

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
- O uso de mais de uma fonte aumenta a diversidade visual e reduz a dependência de um único domínio. O conjunto de teste permanece sempre separado, para medir a capacidade de generalização.
- Para usar todas as imagens do Indoor, em vez da amostra: `python scripts/prepare_dataset.py --sources indoor --max-images 0`.

---

## 4. Metodologia

### 4.1 Preparação dos dados

Download automático, normalização das anotações para o formato YOLO, eliminação de duplicatas e divisão em treino, validação e teste.

### 4.2 Treinamento

Fine-tuning do YOLO com data augmentation nativo do Ultralytics (flip, escala, mosaic, mixup, entre outros).

| Configuração | Modo acadêmico leve (padrão) | Treinamento completo |
|---|---|---|
| Dataset | Indoor, 1.500 imagens | D-Fire |
| Épocas | 10 | 30 |
| Resolução (`imgsz`) | 512 | 640 |
| Batch | 4 | 4 |

### 4.3 Pré-processamento

O módulo `detector/preprocess.py` é configurável e independente do modelo, o que permite comparar a imagem original com a processada: resize, Gaussian Blur, denoise, ajuste de contraste e conversões BGR → RGB e BGR → HSV. O projeto não pressupõe que cada transformação melhore o desempenho; isso deve ser verificado experimentalmente.

### 4.4 Experimento de thresholds

O modelo é avaliado com limiares de confiança `0.25`, `0.40`, `0.50`, `0.60` e `0.75`, registrando Precision, Recall, mAP50 e mAP50-95. O objetivo é evidenciar o compromisso entre falsos positivos e falsos negativos.

---

## 5. Instalação e execução

**Requisitos:** Python 3.10 ou superior (exigido pelo SDK `groq`) e conexão com a internet para o download dos datasets.

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

O modelo padrão é o `openai/gpt-oss-120b`, acessado pela API da Groq e configurável por meio da variável `FIREWATCH_GROQ_MODEL`.

**Funcionalidades**

- **Relatórios Inteligentes:** relatório técnico em português com resumo executivo, estatísticas por classe, análise temporal, pontos de atenção, recomendações de verificação humana e conclusão, exportável em TXT e PDF.
- **Assistente Inteligente:** perguntas em linguagem natural sobre o histórico de detecções.
- **Resumo automático:** na página de Monitoramento, ao concluir o processamento de um vídeo, é gerado um relatório apenas dos registros criados naquela execução. A opção vem ativada por padrão e realiza uma chamada à API por vídeo com novos registros; desmarque-a para não consumir tokens. Se não houver detecções ou a API estiver indisponível, o vídeo permanece processado e o histórico é preservado.

**Privacidade e confiabilidade**

- Somente metadados e estatísticas são enviados à Groq; imagens, vídeos, caminhos locais e coordenadas nunca são transmitidos.
- Os números dos relatórios são calculados em Python, e o modelo de linguagem apenas redige a interpretação.
- A resposta é exigida em JSON Schema estrito.

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
├── ai/                         Prompts e geração de relatórios
├── detector/                   Detecção YOLO e pré-processamento
├── interface/                  Páginas: home, monitor, webcam, history, ai_reports, settings
├── utils/                      Vídeo, logger e alertas
├── scripts/                    Download, preparo, treino, validação e experimentos
├── data_sources/               Datasets brutos (gerado)
├── datasets/fire_smoke/        Dataset normalizado (gerado)
├── models/                     Pesos treinados (gerado)
├── runs/ e reports/            Saídas de treino e avaliação (gerado)
├── docs/                       Documentação complementar
├── tests/                      Testes (test_ai_reports.py)
└── screenshots/, videos/, logs/
```

O script `scripts/create_synthetic_dataset.py` é uma ferramenta didática e experimental, e não faz parte do pipeline principal.

---

## 9. Limitações e trabalhos futuros

**Limitações.** Mesmo com datasets reais, o desempenho depende da qualidade e resolução das imagens, do domínio das câmeras, da iluminação, da distância, do tamanho do fogo, da densidade da fumaça e do ambiente (interno ou externo). Um bom resultado no conjunto de teste não garante segurança em ambiente real.

**Trabalhos futuros.**

- Treinar com o conjunto completo e combinar múltiplas fontes de dados.
- Avaliar o efeito de cada etapa de pré-processamento sobre as métricas.
- Testar o modelo em vídeos externos, de domínio distinto do treinamento.
- Comparar diferentes variantes do YOLO.

---

## 10. Referências

- Venâncio, P. V. A. B.; Lisboa, A. C.; Barbosa, A. V. *An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices*. Neural Computing and Applications, 2022. Dataset D-Fire: [github.com/gaia-solutions-on-demand/DFireDataset](https://github.com/gaia-solutions-on-demand/DFireDataset).
- Putra, A. K. *Indoor Fire Smoke Dataset*. Zenodo, 2025. DOI [10.5281/zenodo.15826133](https://doi.org/10.5281/zenodo.15826133).

---

## 11. Integrantes

- Felipe Piovesan
- Frederico Faben Brumati
- Gustavo Vasconcelos Solomão
- Ruan Gimenes
- Vitor do Nascimento Zuchieri
- José Roberto Miranda Neto
