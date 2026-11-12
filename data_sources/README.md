# Fontes de dados

Os arquivos brutos não são versionados no projeto. Os scripts fazem o download automaticamente.

## D-Fire

Dataset público com mais de 21 mil imagens de fogo/fumaça e bounding boxes em formato YOLO.

Fonte original: https://github.com/gaia-solutions-on-demand/DFireDataset

## Indoor Fire Smoke Dataset

Dataset de 5.000 imagens reais de ambientes internos com bounding boxes para fogo e fumaça.

DOI: https://doi.org/10.5281/zenodo.15826133

Para usar os dois:

```powershell
python scripts/run_pipeline.py --sources dfire indoor --epochs 50
```
