"""
config.py

Configuración general para el módulo de corrección geométrica y coregistro multiespectral.

Este archivo define las rutas base del proyecto y permite iterar sobre las distintas zonas de estudio ubicadas
dentro de la carpeta `data/Input/`. También controla si se deben limpiar los resultados previos en cada ejecución.
"""

import os
from pathlib import Path

INPUT_DIR = Path(r"D:\preprocesamiento_2025\T2\0_IMG_sin_redundancia")
OUTPUT_DIR = Path(r"D:\preprocesamiento_2025\T2")

# Obtener la ruta base del script actual
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta de la carpeta donde están las imágenes de entrada
#INPUT_DIR = os.path.join(BASE_DIR, "data", "Input")

# Ruta de la carpeta donde se guardarán las imágenes procesadas, organizadas por zona de estudio
#OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Output")

# Subcarpetas que se crearán dentro de cada zona de estudio al procesar
SUBCARPETA_CORRECCION = "correccion_geometrica"
SUBCARPETA_COEGISTRO = "coregistro"

# Opción para borrar resultados previos antes de procesar cada zona
LIMPIAR_OUTPUT_ANTES_DE_PROCESAR = False


def listar_zonas_disponibles():
    """
    Devuelve una lista de nombres de carpetas (zonas de estudio) dentro de `data/Input/`.
    """
    if not os.path.exists(INPUT_DIR):
        print("La carpeta de entrada no existe:", INPUT_DIR)
        return []
    
    return [nombre for nombre in os.listdir(INPUT_DIR)
            if os.path.isdir(os.path.join(INPUT_DIR, nombre))]
