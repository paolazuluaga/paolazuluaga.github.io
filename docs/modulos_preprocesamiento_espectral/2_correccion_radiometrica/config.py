"""
config.py

Configuración del módulo de main_radiometrica para la corrección radiométrica de las imágenes multiespectrales.

Define las rutas de entrada y salida para facilitar la ejecución del script sin necesidad de modificar el código fuente.
"""


import os

# Obtener la ruta base del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta de la carpeta donde están las imágenes de entrada
INPUT_DIR = os.path.join(BASE_DIR, "data", "Input")

# Ruta de la carpeta donde se encuentra el archivo csv con los metadatos de las imágenes
METADATA_CSV = os.path.join(INPUT_DIR, "metadata_tif.csv")

# Ruta de la carpeta donde se guardarán las imágenes procesadas, organizadas por zona de estudio
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "Output")

# Opción para borrar resultados previos antes de procesar cada zona
LIMPIAR_OUTPUT_ANTES_DE_PROCESAR = True
