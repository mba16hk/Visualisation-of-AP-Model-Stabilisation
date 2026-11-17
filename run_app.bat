@echo off
rem Start the stabilisation results browser and open it in the browser.
rem Uses the Windows Python launcher's newest Python 3 if present, else "python".
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
start "" http://localhost:8501
%PY% -m streamlit run app.py
