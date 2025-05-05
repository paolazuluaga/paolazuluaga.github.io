"""
MÓDULO PRINCIPAL DE CORRECCIÓN GEOMÉTRICA Y COREGISTRO MULTIESPECTRAL

Autor: Paola Andrea Mejia-Zuluaga
Fecha: Abril 21 de 2025  
Proyecto: Preprocesamiento de imágenes para el Proyecto - Monitoreo de especies de Muérdago  
          en Parques Urbanos Usando Imágenes Aéreas e Inteligencia Artificial  
Versión: 1.0  
Contacto: paomejia23@gmail.com  

Descripción:
Este script ejecuta el flujo completo de preprocesamiento geométrico de imágenes multiespectrales y RGB
capturadas con sensores UAV. Abarca tres etapas principales:

1. **Corrección geométrica individual**  
   Aplica corrección de distorsión de lente y homografía en función de los parámetros de calibración de cada cámara,
   extraídos desde metadatos técnicos. Se utiliza una transformación afín y perspectiva para corregir el paralaje
   y las inclinaciones (roll, pitch, yaw).

2. **Coregistro espectral multibanda**  
   Alinea espacialmente las bandas multiespectrales (1 a 4) con respecto a la banda NIR (banda 5), combinando:
   - Corrección inicial basada en los desplazamientos de los centros ópticos desde los metadatos.
   - Refinamiento con detección de keypoints y homografía mediante SIFT (Scale-Invariant Feature Transform).

3. **Generación de imagen RGB transferida**  
   Crea una imagen RGB realista a partir de las bandas 3-2-1 del TIF multibanda, ajustando su apariencia al estilo
   fotográfico del JPG mediante transferencia de color (método de Reinhard).

Cada zona de estudio se procesa de forma independiente. Si no se encuentran archivos de metadatos (`metadata_tif.csv` 
y `metadata_jpg.csv`), estos se generan automáticamente mediante el módulo `metadata_extractor.py` usando ExifTool.

Entradas:
- `data/Input/[zona]/`: Carpeta con las imágenes originales (.TIF y .JPG) de cada zona.
- Metadatos (`metadata_tif.csv`, `metadata_jpg.csv`) si ya existen.

Salidas:
- `data/Output/[zona]/correccion_geometrica/`: Imágenes corregidas geométricamente.
- `data/Output/[zona]/coregistro/`: Imágenes coregistradas multibanda y RGB transferidas.
- Archivos `progreso.txt` y `progreso_coregistro.txt` que registran el avance del procesamiento por imagen.

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
- Estructura modular compatible con `config.py` y scripts auxiliares en `scripts/`.

Uso:
    python main_correccion_geometrica.py

Notas:
- Se recomienda ejecutar este módulo antes de la etapa de corrección radiométrica o clasificación.
- El código está diseñado para ser fácilmente integrado en flujos de trabajo por lotes y reproducibles.
"""

# =============================== IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN ===============================
import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import shutil
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, SUBCARPETA_CORRECCION, SUBCARPETA_COEGISTRO, listar_zonas_disponibles
from scripts.metadata_extractor import generate_metadata_csv
from scripts.utils import correct_lens_distortion, crop_center, extract_camera_params, save_corrected_image, save_processed_image
from scripts.coregistro import ejecutar_coregistro
from scripts.generar_rgb import generar_rgb_transferido_batch



# =============================== FUNCIÓN: CORRECCIÓN GEOMÉTRICA DE UNA IMAGEN ===============================
def corregir_imagen(image_path, metadata, output_folder, tipo_imagen="TIF", progress_file="progreso.txt"):
    """
    Aplica la corrección geométrica a una imagen aérea multiespectral (TIF) o RGB (JPG) a partir de sus metadatos de calibración.

    Esta función realiza tres pasos fundamentales para cada imagen:
    1. Corrección de distorsión de lente con los parámetros de la cámara.
    2. Corrección de perspectiva o paralaje mediante homografía, si hay inclinación significativa (roll, pitch, yaw).
    3. Recorte centrado para normalizar el tamaño final a un ROI definido (1500x1150 px).

    El proceso se aplica tanto a imágenes multiespectrales (`.TIF`) como RGB (`.JPG`), utilizando metadatos previamente
    extraídos. La imagen corregida se guarda en una carpeta de salida y su nombre se registra en un archivo
    de progreso para evitar reprocesamiento en ejecuciones futuras.

    Args:
        image_path (str): Ruta completa del archivo de imagen a corregir (TIF o JPG).
        metadata (pd.DataFrame): DataFrame con los metadatos extraídos vía ExifTool (uno por tipo de imagen).
        output_folder (str): Carpeta donde se guardará la imagen corregida geométricamente.
        tipo_imagen (str): Tipo de imagen a procesar. Puede ser 'TIF' (multiespectral) o 'JPG' (RGB).
        progress_file (str): Ruta del archivo de texto donde se registran los nombres de imágenes ya procesadas.

    Returns:
        None. Guarda directamente la imagen corregida en el disco.

    Notas:
        - Si la imagen ya ha sido procesada previamente (aparece en `progress_file`), se omite automáticamente.
        - La corrección geométrica por homografía solo se aplica si alguno de los valores de roll, pitch o yaw supera ±0.1.
        - Las imágenes TIF se guardan con su perfil rasterio original actualizado (multibanda o monobanda).
        - Las imágenes JPG se guardan en formato comprimido (`uint8`) utilizando OpenCV.

    Ejemplo de uso:
        corregir_imagen("DJI_0011_ob6_e_t3.TIF", metadatos_tif, "Output/Exconvento/correccion_geometrica", tipo_imagen="TIF")

    Requiere:
        - Funciones auxiliares: `extract_camera_params`, `correct_lens_distortion`, `crop_center`,
          `save_corrected_image`, `save_processed_image`.
        - Metadatos válidos y alineados con el nombre exacto del archivo (`File Name` en el CSV).
    """

    image_name = os.path.basename(image_path)
    meta_row = metadata[metadata['File Name'].str.strip().str.lower() == image_name.lower()]
    if meta_row.empty:
        print(f"Metadatos no encontrados para {image_name}. Omitiendo.")
        return
    meta_row = meta_row.iloc[0]

    # Verificar si la imagen ya ha sido procesada
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_images = set(f.read().splitlines())
        if image_name in processed_images:
            print(f"{image_name} ya fue procesada. Omitiendo.")
            return

    # Cargar la imagen
    if tipo_imagen == "TIF":
        with rasterio.open(image_path) as src:
            img = np.stack([src.read(i + 1) for i in range(src.count)], axis=-1)
            profile = src.profile
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Extraer parámetros de calibración
    camera_matrix, dist_coeffs, homography_matrix = extract_camera_params(meta_row, tipo_imagen)
    if camera_matrix is None:
        return

    # Paso 1: Corrección de distorsión
    undistorted_img, roi = correct_lens_distortion(img, camera_matrix, dist_coeffs)

    # Paso 2: Corrección geométrica
    h, w = undistorted_img.shape[:2]
        # Extraer Roll, Pitch, Yaw
    roll, pitch, yaw = float(meta_row['Roll']), float(meta_row['Pitch']), float(meta_row['Yaw'])
        # Aplicar homografía en todas las imágenes, pero en ortogonales solo si hay inclinación significativa
    if abs(roll) > 0.1 or abs(pitch) > 0.1 or abs(yaw) > 0.1:
        undistorted_img = cv2.warpPerspective(undistorted_img, homography_matrix, (w, h), flags=cv2.INTER_LINEAR)
        print(f"Corrección geométrica aplicada en {image_name} debido a Roll={roll}, Pitch={pitch}, Yaw={yaw}")
    else:
        print(f"No se requiere corrección en la imagen ortogonal {image_name}, sin inclinación significativa.")

    # Paso 3: Recorte al tamaño deseado
    corrected_img = crop_center(undistorted_img)

    # Guardar imagen corregida
    output_path = os.path.join(output_folder, image_name)
    save_corrected_image(corrected_img, output_path, tipo_imagen, profile if tipo_imagen == "TIF" else None)

    # Guardar imagen en progreso.txt
    save_processed_image(progress_file, image_name)

    print(f"Imagen corregida guardada en: {output_path}")



# =============================== FUNCIÓN: PROCESAMIENTO COMPLETO DE UNA ZONA DE ESTUDIO ===============================
def procesar_zona(zona):
    """
    Ejecuta el preprocesamiento completo para una zona de estudio, incluyendo:
    1. Corrección geométrica de imágenes TIF y JPG.
    2. Coregistro espectral multibanda respecto a la banda NIR.
    3. Generación de imágenes RGB visualmente naturalizadas por transferencia de color.

    Este flujo es aplicado a cada carpeta individual dentro del directorio de entrada (`INPUT_DIR`), 
    correspondiente a una zona o sitio de estudio. El procesamiento asegura la generación de productos 
    corregidos geométricamente, alineados espacialmente y listos para análisis espectral o segmentación.

    Pasos del procesamiento:
    - Verifica y/o genera automáticamente los archivos de metadatos (`metadata_tif.csv`, `metadata_jpg.csv`)
      usando `ExifTool`, en caso de que no existan.
    - Aplica corrección geométrica a todas las imágenes .TIF y .JPG usando `corregir_imagen()`.
    - Ejecuta el coregistro multiespectral con `ejecutar_coregistro()`, usando la banda NIR como referencia.
    - Genera una imagen RGB naturalizada a partir de las bandas 3-2-1 del TIF, transferida al estilo del JPG.

    Args:
        zona (str): Nombre de la carpeta de la zona de estudio, ubicada dentro de `data/Input`.

    Returns:
        None. Todos los productos son guardados en disco en subcarpetas dentro de `data/Output/[zona]/`.

    Notas:
        - Las imágenes corregidas geométricamente se almacenan en `Output/[zona]/correccion_geometrica/`.
        - Las imágenes coregistradas (TIF multibanda y RGB transferido) se almacenan en `Output/[zona]/coregistro/`.
        - El archivo de progreso `progreso.txt` evita reprocesamiento redundante por imagen.
        - Si ExifTool no está instalado, el procesamiento se detiene con un mensaje de error informativo.

    Requiere:
        - Librerías: `os`, `pandas`, `shutil`, `cv2`, `rasterio`
        - Funciones auxiliares: `corregir_imagen`, `ejecutar_coregistro`, `generar_rgb_transferido_batch`
        - Script de metadatos: `generate_metadata_csv()` desde `scripts.metadata_extractor.py`
    """

    print(f"\n Procesando zona: {zona}")
    ruta_entrada = os.path.join(INPUT_DIR, zona)
    ruta_corr = os.path.join(OUTPUT_DIR, zona, SUBCARPETA_CORRECCION)
    ruta_coreg = os.path.join(OUTPUT_DIR, zona, SUBCARPETA_COEGISTRO)

    os.makedirs(ruta_corr, exist_ok=True)
    os.makedirs(ruta_coreg, exist_ok=True)

    # Cargar o Extraer metadatos
    metadata_tif_csv = os.path.join(ruta_entrada, 'metadata_tif.csv')
    metadata_jpg_csv = os.path.join(ruta_entrada, 'metadata_jpg.csv')

    if not os.path.exists(metadata_tif_csv) or not os.path.exists(metadata_jpg_csv):
        print(f"\nMetadatos no encontrados en {zona}. Generando automáticamente con ExifTool...")

        # Verificar si ExifTool está disponible en el sistema
        if shutil.which("exiftool") is None:
            print("Error: ExifTool no está instalado o no está en el PATH del sistema.")
            print("Instala ExifTool desde https://exiftool.org/ y asegúrate de que esté accesible por consola.")
            return

        # Generar metadatos
        exito = generate_metadata_csv(ruta_entrada)
        if not exito:
            print(f"No se pudieron generar los metadatos para {zona}. Omitiendo esta zona.")
            return

    # Cargar los metadatos una vez estén asegurados
    metadata_tif = pd.read_csv(metadata_tif_csv)
    metadata_jpg = pd.read_csv(metadata_jpg_csv)

    # Definir ruta del archivo de progreso geométrico por zona
    progress_file = os.path.join(ruta_corr, "progreso.txt")

    # Corrección geométrica de imágenes TIF y JPG
    for fname in os.listdir(ruta_entrada):
        fpath = os.path.join(ruta_entrada, fname)
        if fname.lower().endswith(".tif"):
            corregir_imagen(fpath, metadata_tif, ruta_corr, tipo_imagen="TIF", progress_file=progress_file)
        elif fname.lower().endswith((".jpg", ".jpeg")):
            corregir_imagen(fpath, metadata_jpg, ruta_corr, tipo_imagen="JPG", progress_file=progress_file)


    # Coregistro espectral respecto a NIR
    ejecutar_coregistro(ruta_corr, ruta_corr, ruta_coreg, metadata_tif_csv, metadata_jpg_csv)

    # Generar imagen RGB desde TIF coregistrado y JPG corregido
    generar_rgb_transferido_batch(directorio_tif=ruta_coreg, directorio_jpg=ruta_corr, directorio_salida=ruta_coreg, verbose=True)


# =============================== BLOQUE PRINCIPAL DE EJECUCIÓN ===============================
if __name__ == "__main__":
    zonas = listar_zonas_disponibles()
    if not zonas:
        print("No se encontraron zonas en la carpeta de entrada.")
    for zona in zonas:
        procesar_zona(zona)


