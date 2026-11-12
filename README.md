# 🔥 FireWatch AI — Pipeline de IA para Detecção de Incêndio e Fumaça

Sistema acadêmico de **visão computacional e Deep Learning** para detectar `Fire` e `Smoke` em vídeos e webcam usando **Ultralytics YOLO**.

A versão atual foi preparada para que **não seja necessário coletar imagens manualmente** e usa por padrão uma amostra acadêmica de **1.500 imagens reais**, suficiente para demonstrar o pipeline sem exigir horas de treinamento em CPU. O pipeline baixa datasets públicos de incêndio/fumaça, normaliza as anotações, elimina duplicatas, cria `train/val/test`, treina o YOLO, valida no conjunto de teste e executa experimentos de threshold.

> **Importante:** o ZIP não inclui os datasets brutos nem `fire_model.pt` porque esses arquivos são grandes. O projeto baixa e gera tudo automaticamente com scripts reproduzíveis.

---

## 🎯 Objetivo acadêmico

O projeto demonstra explicitamente um pipeline de Inteligência Artificial:

```text
DATASETS PÚBLICOS
      ↓
COLETA AUTOMÁTICA
      ↓
CURADORIA / DEDUPLICAÇÃO
      ↓
NORMALIZAÇÃO DAS ANOTAÇÕES
      ↓
TRAIN / VALIDATION / TEST
      ↓
DATA AUGMENTATION
      ↓
PRÉ-PROCESSAMENTO
      ↓
YOLO — DEEP LEARNING
      ↓
MÉTRICAS
Precision / Recall / mAP50 / mAP50-95
      ↓
TESTE EXTERNO
      ↓
INFERÊNCIA EM VÍDEO / WEBCAM
      ↓
ALERTA + SCREENSHOT + LOG
      ↓
DASHBOARD
```

Isso permite apresentar o projeto não apenas como uma interface, mas como um **experimento de Machine Learning reproduzível**.

---

# 🤖 Onde está a IA?

O FireWatch AI agora possui **duas camadas de Inteligência Artificial com funções diferentes**:

### 1. YOLO — visão computacional

O YOLO é o modelo de Deep Learning responsável por detectar visualmente duas classes:

- `0 = Fire`
- `1 = Smoke`

Durante a inferência, recebe frames e retorna classe, pontuação de confiança e bounding box. A aplicação usa essas saídas para gerar alertas, screenshots e registros.

### 2. LLM Groq — IA generativa

A página **🧠 Relatórios Inteligentes** usa um modelo de linguagem disponibilizado pela Groq para interpretar os registros já produzidos pelo YOLO e redigir relatórios técnicos em português, com resumo, análise por classe, observações temporais, pontos de atenção, recomendações e conclusão.

A IA generativa não substitui a detecção visual e não recebe imagens ou vídeos nesta integração. Ela trabalha com metadados e estatísticas, e nunca deve ser interpretada como confirmação de incêndio real.

Configuração, arquitetura, prompt, segurança, testes e limitações: [documentação da IA generativa](docs/IA_GENERATIVA_GROQ.md).

---

# 📦 Datasets utilizados

## 1. D-Fire — padrão

O D-Fire possui mais de 21 mil imagens de fogo e fumaça e anotações de bounding boxes em formato YOLO. A distribuição publicada informa 14.692 bounding boxes de Fire e 11.865 de Smoke. citeturn1view0

Fonte original: https://github.com/gaia-solutions-on-demand/DFireDataset

O projeto baixa automaticamente uma cópia pública do dataset e preserva a atribuição da fonte original.

## 2. Indoor Fire Smoke Dataset — opcional

Dataset publicado no Zenodo com **5.000 imagens reais** de ambientes internos, contendo bounding boxes de fogo e fumaça e divisão 70/15/15 para treino, validação e teste. citeturn4view0

DOI: https://doi.org/10.5281/zenodo.15826133

Para usar os dois datasets:

```powershell
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10
```

### Por que usar mais de uma fonte?

A ideia é aumentar a diversidade visual e reduzir dependência de um único domínio. O conjunto de teste deve permanecer separado para medir a capacidade de generalização.

---

# 🚀 Execução automática — recomendada

No Windows, basta executar:

```powershell
setup_and_train.bat
```

O script:

1. cria `.venv`;
2. instala as dependências;
3. baixa o D-Fire;
4. prepara o dataset;
5. treina o YOLO;
6. copia o melhor peso para `models/fire_model.pt`;
7. valida no teste;
8. executa o experimento de thresholds.

Depois:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

---

# 🧪 Execução manual

## Instalar

Use Python **3.10 ou superior** (a biblioteca oficial `groq` atualmente requer Python 3.10+).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Baixar dataset

```powershell
python scripts/download_datasets.py --datasets dfire
```

Ou os dois:

```powershell
python scripts/download_datasets.py --datasets dfire indoor
```

## Preparar dataset

```powershell
python scripts/prepare_dataset.py --sources dfire
```

## Treinar

```powershell
python scripts/train_model.py --epochs 10 --imgsz 512 --batch 4
```

Com GPU NVIDIA:

```powershell
python scripts/train_model.py --epochs 10 --imgsz 512 --batch 4 --device 0
```

## Validar no teste

```powershell
python scripts/validate_model.py
```

## Experimento de threshold

```powershell
python scripts/evaluate_experiments.py
```

O resultado é salvo em:

```text
reports/evaluation_summary.csv
```

## Pipeline completo

```powershell
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10
```

Ou:

```powershell
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10
```

---

# 📊 Experimentos

O pipeline avalia diferentes thresholds:

```text
0.25
0.40
0.50
0.60
0.75
```

E registra:

- Precision
- Recall
- mAP50
- mAP50-95

Isso permite demonstrar o trade-off entre **falsos positivos e falsos negativos**.

O treinamento também usa data augmentation do Ultralytics, incluindo transformações como flip, escala, mosaic e mixup.

---

# 🧠 Processamento de imagens

O FireWatch mantém o módulo de pré-processamento configurável:

- resize;
- Gaussian Blur;
- denoise;
- contraste;
- BGR → RGB;
- BGR → HSV.

O processamento é separado do modelo para permitir comparar uma imagem original com uma versão processada.

> O projeto não afirma que cada transformação melhora o modelo. Isso pode ser avaliado experimentalmente.

---

# 🏗️ Estrutura

```text
firewatch-ai/
├── app.py
├── setup_and_train.bat
├── requirements.txt
├── README.md
│
├── ai/
│   ├── prompts.py
│   └── report_generator.py
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
│   ├── run_pipeline.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── evaluate_experiments.py
│   ├── test_model.py
│   ├── extract_frames.py
│   └── create_synthetic_dataset.py
│
├── data_sources/
├── datasets/fire_smoke/
├── models/
├── runs/
├── reports/
├── docs/
│   └── IA_GENERATIVA_GROQ.md
├── tests/
│   └── test_ai_reports.py
├── screenshots/
├── videos/
└── logs/
```

`create_synthetic_dataset.py` permanece apenas como ferramenta didática/experimental. **Ele não faz parte do pipeline principal.**

---

# 🔬 Critérios para uma boa avaliação

<<<<<<< HEAD
O projeto deve apresentar pelo menos:

- dataset de origem identificado;
- quantidade de imagens;
- quantidade de classes;
- distribuição Train/Validation/Test;
- método de anotação;
- ausência de duplicatas;
- modelo e pesos base;
- épocas de treinamento;
- data augmentation;
- Precision;
- Recall;
- mAP50;
- mAP50-95;
- matriz de confusão gerada pelo YOLO;
- exemplos de verdadeiros positivos;
- exemplos de falsos positivos;
- exemplos de falsos negativos;
- teste em vídeo;
- teste em webcam;
- limitações e trabalhos futuros.

---

# ⚠️ Limitações

O FireWatch AI é um **protótipo acadêmico**. Ele não substitui sistemas certificados de detecção e alarme de incêndio.

Mesmo usando datasets reais, a qualidade final depende de fatores como:

- qualidade das imagens;
- domínio das câmeras utilizadas;
- iluminação;
- distância;
- tamanho do fogo;
- densidade da fumaça;
- resolução;
- ambiente interno/externo.

Um resultado alto no teste não deve ser apresentado como garantia de segurança em ambiente real.

---

# 📚 Referências dos datasets

**D-Fire** — Venâncio, P. V. A. B.; Lisboa, A. C.; Barbosa, A. V. *An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices*. Neural Computing and Applications, 2022. A página oficial do dataset disponibiliza imagens, labels e divisões para treinamento/teste. citeturn1view0

**Indoor Fire Smoke Dataset** — Putra, Arya Krisna. Dataset publicado no Zenodo, 2025, DOI 10.5281/zenodo.15826133. citeturn4view0

---

# 🏁 Resultado esperado

Ao final do pipeline, o projeto deve possuir:

```text
models/fire_model.pt
reports/evaluation_summary.csv
runs/firewatch/train/
reports/data_runtime.yaml
```

E a aplicação poderá executar:

```text
🔥 FireWatch AI
       ↓
YOLO treinado
       ↓
Fire / Smoke
       ↓
Bounding Box + Confidence
       ↓
Alerta
       ↓
Screenshot + CSV
       ↓
Dashboard
```

## Execução automática no Windows

Na raiz do projeto, execute no PowerShell:

```powershell
.\setup_and_train.bat
```

O script cria `.venv`, instala as dependências, baixa o D-Fire automaticamente, prepara o dataset, treina o YOLO, valida no conjunto de teste e executa os experimentos.

> Observação: em Windows com GPU AMD, a instalação padrão do PyTorch/Ultralytics normalmente treina em CPU. O script detecta automaticamente NVIDIA CUDA quando `nvidia-smi` está disponível; caso contrário usa CPU.

## Download robusto de datasets

O pipeline **não exige coleta manual de imagens nem anotação manual**. O D-Fire é um dataset público de detecção de fogo e fumaça com mais de 21 mil imagens e anotações YOLO. O downloader tenta primeiro a cópia pública do Kaggle e usa o espelho do Hugging Face como fallback. O caminho utilizado fica salvo em `data_sources/dfire/.source_path`, evitando novo download em execuções seguintes.

O fallback Hugging Face usa timeout maior, suporte a `hf_xet` e retomada de download. Se o Kaggle já tiver concluído o download, o projeto não baixa o dataset novamente.

### Execução

```powershell
.\setup_and_train.bat
```

Ou, manualmente:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py --datasets dfire
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --sources dfire
.\.venv\Scripts\python.exe scripts\train_model.py --epochs 30 --imgsz 640 --batch 4 --device cpu
.\.venv\Scripts\python.exe scripts\validate_model.py
.\.venv\Scripts\python.exe scripts\evaluate_experiments.py
```

> O D-Fire é um dataset de terceiros. O projeto deve citar o dataset e seu trabalho original no relatório/apresentação.


## Dataset padrão (v2)

O pipeline padrão utiliza o **Indoor Fire Smoke Dataset**, publicado no Zenodo, com 5.000 imagens reais de ambientes internos e anotações de bounding boxes para fogo e fumaça. O arquivo publicado tem aproximadamente 200,5 MB. O pipeline baixa e prepara o dataset automaticamente, sem necessidade de coleta ou anotação manual.

O D-Fire permanece disponível como dataset opcional para um treinamento maior: `python scripts/download_datasets.py --datasets dfire` e `python scripts/prepare_dataset.py --sources dfire`.


## 🧑‍🎓 Modo acadêmico leve

Para facilitar a execução em computadores sem GPU NVIDIA, o pipeline padrão usa **1.500 imagens reais** do dataset Indoor Fire Smoke, mantendo aproximadamente 70% para treino, 15% para validação e 15% para teste. O treinamento padrão usa **10 épocas**, `imgsz=512` e `batch=4`.

Isso não altera o dataset de origem: é apenas uma amostragem reprodutível (`seed=42`) para o experimento acadêmico. Para usar todo o dataset, execute `python scripts/prepare_dataset.py --sources indoor --max-images 0`.

---

## IA generativa — Relatórios Inteligentes (Groq)

A página **🧠 Relatórios Inteligentes** utiliza o modelo `openai/gpt-oss-120b` via API Groq para produzir relatórios técnicos em português com resumo executivo, estatísticas por classe, análise temporal, pontos de atenção, recomendações de verificação humana e conclusão.

A aplicação prepara os dados localmente, envia apenas metadados/estatísticas (nunca imagens, vídeos, caminhos locais ou coordenadas) e exige uma resposta em JSON Schema estrito. Os números apresentados no relatório são calculados pelo Python; a IA gera a interpretação em linguagem natural.

**Configuração e detalhes:** consulte [docs/IA_GENERATIVA_GROQ.md](docs/IA_GENERATIVA_GROQ.md). O documento explica o fluxo YOLO → Groq, prompt, modelos, segurança, limitações, instalação e testes.

Para habilitar a integração, configure `GROQ_API_KEY` no `.env` usando `.env.example` como base. A chave não deve ser publicada no GitHub. O SDK Groq requer Python 3.10 ou superior.

## 💬 Assistente Inteligente e resumo automático

A aplicação inclui a página **Assistente Inteligente**, que permite fazer perguntas em linguagem natural sobre o histórico registrado. O assistente recebe estatísticas agregadas e uma amostra cronológica dos grupos de evidência; caminhos locais, coordenadas, imagens e vídeos não são enviados à Groq.

Na página **Monitoramento**, a opção **Gerar resumo automático com IA ao terminar o vídeo** vem ativada por padrão. Após o processamento, o sistema seleciona somente os novos registros criados naquela execução, gera um relatório técnico usando o mesmo modelo configurado em `FIREWATCH_GROQ_MODEL` e oferece exportação TXT/PDF. Se não houver detecções ou a API estiver indisponível, o vídeo continua processado e o erro de resumo é informado sem apagar o histórico.

A geração automática faz uma chamada à API por vídeo concluído com novos registros. Desmarque a opção para não consumir tokens nessa etapa.
=======
O modelo atual é um **protótipo acadêmico**. Ele não deve ser considerado um sistema certificado de segurança contra incêndios nem substituir equipamentos ou sistemas profissionais de detecção e alarme.

## 👥 Integrantes

- Felipe Piovesan
- Frederico Faben Brumati
- Gustavo Vasconcelos Solomão
- Ruan Gimenes
- Vitor do Nascimento Zuchieri
- José Roberto Miranda Neto
>>>>>>> 0ffc7e6ae84e1caa31564635903330aa3379b0da
