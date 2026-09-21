# 🔥 FireWatch AI — Detecção de Incêndio e Fumaça com Inteligência Artificial

Sistema acadêmico de **Visão Computacional e Deep Learning** para detecção de **fogo (Fire)** e **fumaça (Smoke)** em imagens, vídeos e webcam utilizando **Ultralytics YOLO**.

O projeto foi estruturado como um **pipeline de Inteligência Artificial reproduzível**, desde a obtenção automática de datasets públicos até treinamento, avaliação e inferência em vídeo.

---

## 🎯 Objetivo

Desenvolver e demonstrar um sistema de detecção automática de incêndio e fumaça utilizando um modelo de Deep Learning, avaliando seu desempenho por métricas objetivas e integrando o modelo a uma aplicação de monitoramento.

O projeto não depende de coleta manual de imagens ou anotação manual para o fluxo principal: os datasets são obtidos automaticamente, preparados e convertidos para o formato utilizado pelo YOLO.

---

# 🤖 Pipeline de Inteligência Artificial

```text
┌──────────────────────────────┐
│      DATASETS PÚBLICOS       │
│ Indoor Fire Smoke / D-Fire   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     DOWNLOAD AUTOMÁTICO      │
│        scripts/              │
│    download_datasets.py      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│   CURADORIA E PREPARAÇÃO     │
│ deduplicação + normalização  │
│        das anotações         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      DATASET YOLO            │
│ Fire = 0 | Smoke = 1         │
│ Train / Validation / Test    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      DATA AUGMENTATION        │
│ flip / scale / mosaic / etc. │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│        TREINAMENTO            │
│       YOLO + Deep Learning    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│          AVALIAÇÃO            │
│ Precision / Recall / mAP50    │
│          mAP50-95             │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       TESTE EXTERNO           │
│ imagens / vídeos não usados   │
│       no treinamento          │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       INFERÊNCIA              │
│ vídeo / webcam / imagem       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ ALERTA + SCREENSHOT + LOG     │
└──────────────────────────────┘
```

---

# 🧠 Onde está a Inteligência Artificial?

A IA está no modelo **YOLO (You Only Look Once)**, utilizado como detector de objetos.

O modelo foi treinado para reconhecer duas classes:

```text
0 → Fire
1 → Smoke
```

Para cada objeto detectado, o modelo produz:

- classe;
- nível de confiança (confidence);
- bounding box.

Exemplo:

```text
Fire
Confidence: 0.949
Bounding Box: x1, y1, x2, y2
```

A aplicação utiliza essas predições para gerar a visualização e, quando uma detecção atende ao limiar configurado, gerar o alerta de fogo/fumaça.

---

# 📦 Dataset

## Dataset principal — Indoor Fire Smoke

A configuração acadêmica leve utiliza o **Indoor Fire Smoke Dataset**, com 5.000 imagens reais disponíveis.

Para reduzir o custo computacional do treinamento, o pipeline pode trabalhar com uma amostra reprodutível de até **1.500 imagens**, selecionadas automaticamente com `seed=42`.

A divisão planejada é aproximadamente:

```text
70% → treinamento
15% → validação
15% → teste
```

A seleção e a preparação são feitas automaticamente por:

```text
scripts/prepare_dataset.py
```

## Dataset adicional — D-Fire

O projeto também possui suporte ao **D-Fire**, um dataset público voltado à detecção de fogo e fumaça.

Ele pode ser utilizado em experimentos posteriores para aumentar a diversidade dos dados.

---

# 🔄 Preparação dos dados

O script de preparação executa:

1. localização das imagens;
2. localização das anotações;
3. leitura das bounding boxes;
4. normalização das classes;
5. conversão para o formato YOLO quando necessário;
6. remoção de imagens duplicadas por hash SHA-256;
7. divisão em treino, validação e teste;
8. geração do `data.yaml`;
9. geração do `dataset_manifest.json`.

No FireWatch:

```text
Fire  → classe 0
Smoke → classe 1
```

No caso do D-Fire, a convenção de classes é normalizada automaticamente pelo pipeline.

---

# 🧪 Treinamento

O treinamento utiliza **Ultralytics YOLO** com pesos pré-treinados, realizando transferência de aprendizado (transfer learning).

Configuração acadêmica utilizada:

```text
Épocas:        10
Imagem:        512 × 512
Batch:         4
Workers:       2
Patience:      5
```

Durante o treinamento são utilizadas técnicas de data augmentation, incluindo transformações como:

- flip horizontal;
- escala;
- translação;
- rotação;
- mosaic;
- mixup.

O melhor modelo é salvo em:

```text
runs/firewatch/train/weights/best.pt
```

e copiado para:

```text
models/fire_model.pt
```

---

# 📊 Avaliação

O pipeline calcula:

- **Precision**
- **Recall**
- **mAP@50**
- **mAP@50-95**

Também são testados diferentes thresholds de confiança:

```text
0.25
0.40
0.50
0.60
0.75
```

Isso permite observar o equilíbrio entre falsos positivos e falsos negativos.

O resultado é salvo em:

```text
reports/evaluation_summary.csv
```

## Resultado registrado no treinamento acadêmico

| Confidence | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.25 | 51,94% | 48,64% | **36,68%** | **15,74%** |
| 0.40 | 53,97% | 46,99% | 32,10% | 13,53% |
| 0.50 | 59,27% | 37,06% | 27,05% | 11,20% |
| 0.60 | 60,84% | 32,09% | 23,68% | 9,97% |
| 0.75 | **67,76%** | 20,71% | 16,05% | 6,79% |

O threshold de 0,25 apresentou o maior mAP50 entre os valores avaliados. Já thresholds maiores aumentaram a Precision, mas reduziram o Recall.

---

# 📈 Artefatos gerados

O treinamento gera automaticamente:

```text
runs/firewatch/train/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxPR_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── train_batch*.jpg
└── val_batch*_labels.jpg
    val_batch*_pred.jpg
```

Esses arquivos permitem analisar visualmente o comportamento do modelo e documentar o experimento.

---

# 🎥 Inferência em vídeo

Depois do treinamento, o modelo pode ser utilizado para processar vídeos.

Fluxo:

```text
Vídeo
  ↓
Extração dos frames
  ↓
Pré-processamento
  ↓
YOLO
  ↓
Detecção Fire / Smoke
  ↓
Confidence
  ↓
Bounding Box
  ↓
Alerta
  ↓
Log / Screenshot
```

Em um teste realizado com vídeo real, a aplicação apresentou aproximadamente:

```text
FPS: 15,6
Detecções: 2
Fire: 82,5%
Fire: 94,9%
```

O sistema exibiu o alerta:

```text
ALERTA — FOGO DETECTADO
```

Esse resultado demonstra a integração entre o modelo de IA e a aplicação de monitoramento.

> O FPS observado é uma medição de demonstração em ambiente local e não deve ser tratado como benchmark universal do sistema.

---

# 🖥️ Aplicação

A interface foi construída com **Streamlit** e permite utilizar o modelo treinado para monitoramento.

Para iniciar:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

O sistema pode trabalhar com:

- imagens;
- vídeos;
- webcam;
- visualização das detecções;
- confidence;
- bounding boxes;
- alertas;
- registro de eventos.

---

# ⚙️ Execução automática

No Windows:

```powershell
.\setup_and_train.bat
```

O pipeline executa:

```text
1. Criar ambiente virtual
2. Instalar dependências
3. Baixar dataset
4. Preparar dataset
5. Treinar YOLO
6. Validar modelo
7. Avaliar thresholds
8. Gerar fire_model.pt
```

Depois:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

# 🧩 Execução manual

## Criar ambiente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Baixar dataset

```powershell
python scripts/download_datasets.py --datasets indoor
```

Para D-Fire:

```powershell
python scripts/download_datasets.py --datasets dfire
```

## Preparar

```powershell
python scripts/prepare_dataset.py --sources indoor --max-images 1500
```

## Treinar

```powershell
python scripts/train_model.py --epochs 10 --imgsz 512 --batch 4
```

## Validar

```powershell
python scripts/validate_model.py
```

## Avaliar thresholds

```powershell
python scripts/evaluate_experiments.py
```

## Executar pipeline completo

```powershell
python scripts/run_pipeline.py --sources indoor --max-images 1500 --epochs 10 --imgsz 512 --batch 4
```

---

# 📁 Estrutura do projeto

```text
FireWatch/
├── app.py
├── setup_and_train.bat
├── requirements.txt
├── README.md
│
├── detector/
│   ├── detector.py
│   └── preprocess.py
│
├── interface/
│   ├── home.py
│   ├── monitor.py
│   ├── settings.py
│   ├── webcam.py
│   └── history.py
│
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── evaluate_experiments.py
│   ├── run_pipeline.py
│   ├── extract_frames.py
│   └── test_model.py
│
├── datasets/
├── data_sources/
├── models/
├── reports/
├── runs/
├── logs/
├── videos/
└── screenshots/
```

---

# 🎓 Contribuição acadêmica

O FireWatch pode ser apresentado como um projeto de **Visão Computacional aplicada à prevenção de incêndios**, contemplando:

1. problema de detecção;
2. obtenção de dados reais;
3. preparação e curadoria do dataset;
4. aplicação de Deep Learning;
5. treinamento por transferência de aprendizado;
6. avaliação quantitativa;
7. análise de threshold;
8. teste com dados externos;
9. inferência em vídeo;
10. geração de alertas.

O diferencial do projeto é a integração entre **pipeline de Machine Learning e aplicação funcional**, permitindo demonstrar tanto o desenvolvimento do modelo quanto sua utilização prática.

---

## 📌 Observação

O modelo atual é um **protótipo acadêmico**. Ele não deve ser considerado um sistema certificado de segurança contra incêndios nem substituir equipamentos ou sistemas profissionais de detecção e alarme.
