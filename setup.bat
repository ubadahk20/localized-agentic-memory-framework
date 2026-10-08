@echo off

echo ==========================================
echo Localized Agentic Memory Framework Setup
echo ==========================================

echo.
echo Creating virtual environment...
python -m venv .venv

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Checking Ollama...
ollama --version

echo.
echo Pulling required AI models...
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Run the application using:
echo run.bat
echo.
pause