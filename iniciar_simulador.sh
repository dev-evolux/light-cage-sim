#!/usr/bin/env bash
set -euo pipefail

# Cambiar al directorio donde está este script
cd "$(dirname "$0")"

# 1. Verificar Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python3 no encontrado."
    echo "Instálalo desde https://www.python.org/ o con: brew install python"
    exit 1
fi
echo "Python: $(python3 --version)"

# 2. Crear entorno virtual si no existe
if [ ! -f "venv/bin/activate" ]; then
    echo "[1/3] Creando entorno virtual local (esto tomará un momento)..."
    python3 -m venv venv
fi

# 3. Activar entorno virtual
echo "[2/3] Activando entorno virtual..."
source venv/bin/activate

# 4. Instalar/verificar dependencias
echo "[3/3] Verificando e instalando librerías necesarias..."
pip install --upgrade pip setuptools wheel --quiet
pip install -r requirements.txt --quiet

echo ""
echo "======================================================="
echo "Todo listo! Iniciando el Simulador Evolux..."
echo "Por favor, NO cierres esta ventana mientras usas la app."
echo "======================================================="
echo ""

# 5. Abrir el navegador luego de 3 segundos (en paralelo)
(sleep 3 && open http://localhost:5001) &

# 6. Iniciar el servidor Flask
python app_sim.py
