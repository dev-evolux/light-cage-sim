@echo off
title Simulador Evolux - Lanzador

:: 1. Cambiar al directorio donde esta este archivo
cd /d "%~dp0"

:: 2. Verificar si Python esta instalado en el computador
python --version >nul 2>&1
if errorlevel 1 goto error_python

:: 3. Crear un entorno virtual si no existe
if exist "venv\Scripts\activate.bat" goto activar_venv
echo [1/3] Creando entorno virtual local por primera vez (esto tomara un momento)...
python -m venv venv
if errorlevel 1 goto error_venv

:activar_venv
:: 4. Activar el entorno virtual
echo [2/3] Activando entorno virtual...
call venv\Scripts\activate.bat

:: 5. Instalar/Verificar dependencias
echo [3/3] Verificando e instalando librerias necesarias...

:: Actualizar herramientas base
python -m pip install --upgrade pip setuptools wheel --quiet

:: Crear un requirements.txt flexible automaticamente (AHORA INCLUYE MATPLOTLIB)
echo Flask^>=3.0.0 > requirements.txt
echo numpy^>=1.26.0 >> requirements.txt
echo scipy^>=1.12.0 >> requirements.txt
echo matplotlib^>=3.8.0 >> requirements.txt

:: Instalar las librerias
pip install -r requirements.txt --quiet
if errorlevel 1 goto error_pip

:: 6. Lanzar la aplicacion
echo.
echo =======================================================
echo Todo listo! Iniciando el Simulador Evolux...
echo Por favor, NO cierres esta ventana negra mientras usas la app.
echo =======================================================
echo.

:: Lanzar el navegador en un proceso paralelo con 3 segundos de retraso
start cmd /c "ping 127.0.0.1 -n 4 > nul & start http://localhost:5001"

:: Iniciar el servidor Flask
python app_sim.py

:: Si el servidor de Python se detiene o falla, la ventana se pausara aqui
pause
goto fin

:error_python
color 0C
echo =======================================================
echo ERROR: No se encontro Python en este computador.
echo Por favor, instala Python desde https://www.python.org/
echo Asegurate de marcar la casilla "Add Python to PATH" durante la instalacion.
echo =======================================================
pause
goto fin

:error_venv
color 0C
echo =======================================================
echo ERROR: Fallo la creacion del entorno virtual.
echo Asegurate de tener permisos de escritura en esta carpeta.
echo =======================================================
pause
goto fin

:error_pip
color 0C
echo =======================================================
echo ERROR: Fallo la instalacion de las librerias.
echo Comprueba tu conexion a internet o la version de Python.
echo =======================================================
pause
goto fin

:fin