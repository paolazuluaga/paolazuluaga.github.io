"""
CÓDIGO DE CORRECCIÓN RADIOMÉTRICA

Autor: Paola Andrea Mejia-Zuluaga  
Fecha: Marzo 25 de 2025  
Proyecto: Preprocesamiento de imágenes para el Proyecto - Monitoreo de especies de Muérdago  
          en Parques Urbanos Usando Imágenes Aéreas e Inteligencia Artificial  
Versión: 1.0  
Contacto: paomejia23@gmail.com  

Descripción:  
Este script implementa la corrección radiométrica de imágenes multiespectrales  
capturadas por dron, aplicando las siguientes correcciones a partir de los metadatos de la cámara:  
1. Sustracción del nivel negro  
2. Calibración del sensor  
3. Conversión a reflectancia  
4. Corrección por viñeteo  
5. Normalización al rango [0, 1]  

El objetivo es preparar las imágenes para análisis espectral y clasificación, conservando la coherencia  
radiométrica entre vuelos.

Requisitos:  
- Python 3.8+  
- Instalar las dependencias con:  
  pip install -r requirements.txt
- Instalar `ExifTool` manualmente:
  **Windows**: Descargar desde [https://exiftool.org/](https://exiftool.org/)
  **Mac/Linux**:
  ```
  sudo apt install libimage-exiftool-perl  # Ubuntu/Debian
  brew install exiftool  # MacOS
  ```

Entradas requeridas:
- Carpeta con imágenes multiespectrales en formato .TIF
- Archivo `metadata_exif.csv` con los metadatos extraídos previamente

Salida:
- Carpeta con las imágenes corregidas en formato .TIF (float32)

Configuración de rutas:
- Las rutas de entrada, salida y metadatos pueden definirse en el archivo `config.py`
"""


import sys
import os
# Añadir carpeta 'scripts/' al path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import pandas as pd
import shutil
from config import INPUT_DIR, OUTPUT_DIR, LIMPIAR_OUTPUT_ANTES_DE_PROCESAR
from corr_radiometrica import process_image_radiometrica
from metadata_extractor import generate_metadata_csv



def ejecutar_correccion_radiometrica():
    print(f"\nBuscando zonas de estudio en: {INPUT_DIR}")

    zonas = sorted([f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))])
    if not zonas:
        print("No se encontraron carpetas dentro de 'Input/'.")
        return

    for zona in zonas:
        carpeta_zona = os.path.join(INPUT_DIR, zona)
        carpeta_salida = os.path.join(OUTPUT_DIR, zona)
        metadata_path = os.path.join(carpeta_zona, "metadata_tif.csv")

        if not os.path.exists(metadata_path):
            print(f"No se encontró 'metadata_tif.csv' en {zona}. Generando metadatos...")
            exito = generate_metadata_csv(carpeta_zona)
            if not exito:
                print(f"No fue posible generar los metadatos para {zona}.")
                continue

        if LIMPIAR_OUTPUT_ANTES_DE_PROCESAR and os.path.exists(carpeta_salida):
            print(f"Eliminando resultados previos en {carpeta_salida}...")
            shutil.rmtree(carpeta_salida)
        os.makedirs(carpeta_salida, exist_ok=True)

        print(f"\nProcesando zona: {zona}")
        metadata = pd.read_csv(metadata_path)
        imagenes = [f for f in os.listdir(carpeta_zona) if f.lower().endswith('.tif')]

        if not imagenes:
            print(f"No se encontraron imágenes .TIF en {zona}.")
            continue

        for imagen in imagenes:
            ruta_imagen = os.path.join(carpeta_zona, imagen)
            process_image_radiometrica(ruta_imagen, metadata, carpeta_salida)
        print(f"Zona {zona} procesada con éxito.\n")
    print("Corrección radiométrica finalizada para todas las zonas.")


if __name__ == "__main__":
    ejecutar_correccion_radiometrica()
