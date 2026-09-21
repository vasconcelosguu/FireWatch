@echo off
setlocal
cd /d "%~dp0"

echo ===============================================
echo        FIREWATCH AI - SETUP + TREINAMENTO
echo ===============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Criando ambiente virtual...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo ERRO: Python 3 nao foi encontrado. Instale Python 3.11 ou 3.12 e tente novamente.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Ambiente virtual ja existe.
)

call ".venv\Scripts\activate.bat"

 echo [2/4] Instalando/atualizando dependencias...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

set DEVICE=cpu
where nvidia-smi >nul 2>&1
if %errorlevel%==0 set DEVICE=0

echo.
echo [3/4] Baixando datasets publicos e preparando o dataset YOLO...
echo Dataset padrao: Indoor Fire Smoke (5.000 imagens reais, ~200 MB)
echo O D-Fire completo continua disponivel como opcao, mas nao e baixado por padrao.
python scripts\download_datasets.py --datasets indoor
if errorlevel 1 goto :error
python scripts\prepare_dataset.py --sources indoor --max-images 1500
if errorlevel 1 goto :error

echo.
echo [4/4] Treinando e avaliando o modelo em escala academica...
echo Dataset de treino: 1.050 imagens (amostra reprodutivel)
echo Epocas: 10
echo Imagem: 512px
echo Dispositivo: %DEVICE%
python scripts\train_model.py --epochs 10 --imgsz 512 --batch 4 --device %DEVICE%
if errorlevel 1 goto :error
python scripts\validate_model.py
if errorlevel 1 goto :error
python scripts\evaluate_experiments.py
if errorlevel 1 goto :error

echo.
echo ===============================================
echo           PIPELINE CONCLUIDO COM SUCESSO
 echo ===============================================
echo Modelo: models\fire_model.pt
echo Resultados: reports\
echo.
echo Para abrir a aplicacao:
echo .venv\Scripts\python.exe -m streamlit run app.py
pause
exit /b 0

:error
echo.
echo ===============================================
echo ERRO DURANTE O PIPELINE
 echo ===============================================
echo Veja a mensagem acima para identificar a etapa que falhou.
pause
exit /b 1
