# Modelos

O modelo treinado é gerado automaticamente em `models/fire_model.pt`.

Ele não é incluído no ZIP para evitar um arquivo binário pesado e para manter o treinamento reproduzível.

```powershell
python scripts/run_pipeline.py --sources dfire --epochs 50
```
