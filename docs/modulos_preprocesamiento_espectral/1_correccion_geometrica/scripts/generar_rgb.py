"""
CÓDIGO PARA GENERAR IMÁGENES RGB BASADAS EN EL COREGISTRO DE LAS BANDAS MULTIESPECTRALES

Módulo para la generación de imágenes RGB con apariencia natural a partir de imágenes multiespectrales coregistradas.

Autor: Paola Andrea Mejia-Zuluaga  
Fecha: Abril 21 de 2025  
Proyecto: Preprocesamiento de imágenes para el Proyecto - Monitoreo de especies de Muérdago  
          en Parques Urbanos Usando Imágenes Aéreas e Inteligencia Artificial  
Versión: 1.0  
Contacto: paomejia23@gmail.com  

Descripción:
Este módulo genera imágenes RGB naturalizadas a partir de imágenes TIF multibanda previamente coregistradas.
Para lograr una apariencia visual coherente con la percepción humana, se aplica una técnica de transferencia
de color basada en Reinhard et al. (2001), que ajusta la representación RGB generada desde las bandas 3-2-1
(R-G-B) del TIF para imitar la distribución cromática de la imagen JPG fotográfica correspondiente.

La imagen resultante es adecuada para:
- Segmentación visual y generación de máscaras.
- Inspección visual asistida.
- Modelos que requieren entrada en formato RGB natural.

Funciones principales:
- `extraer_rgb_desde_tif`: Extrae las bandas R, G, B desde un TIF multibanda.
- `normalizar_rgb_conjuntamente`: Escala linealmente los canales RGB al rango [0, 255].
- `transfer_color_reinhard`: Ajusta el estilo cromático del RGB-TIF para que coincida con el JPG.
- `generar_rgb_transferido_batch`: Aplica el proceso a todas las imágenes de un directorio.

Requisitos:
- Python 3.8+
- Librerías: `opencv-python`, `numpy`, `rasterio`, `Pillow`

Ejemplo de uso:
    generar_rgb_transferido_batch(
        directorio_tif="data/Output/Jardin/coregistro",
        directorio_jpg="data/Output/Jardin/correccion_geometrica",
        directorio_salida="data/Output/Jardin/coregistro"
    )

Notas:
- Este módulo debe ejecutarse tras el `coregistro.py`, una vez las bandas multiespectrales hayan sido alineadas.
- Las imágenes JPG deben estar previamente corregidas geométricamente y conservar su resolución original.
"""

# =============================== IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN ===============================
import os
import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from PIL import Image



# ================== FUNCIONES DE EXTRACCIÓN Y NORMALIZACIÓN RGB ==================
def extraer_rgb_desde_tif(path):
    """
    Extrae las tres primeras bandas (1, 2 y 3) de una imagen TIF multibanda y las reorganiza en el orden RGB.

    Esta función está diseñada para trabajar con imágenes multiespectrales coregistradas, donde se asume que:
    - La banda 1 corresponde al azul (Blue)
    - La banda 2 corresponde al verde (Green)
    - La banda 3 corresponde al rojo (Red)

    Aunque los datos están almacenados originalmente en orden B-G-R, esta función los reorganiza
    y devuelve explícitamente los canales en el orden (R, G, B) para su uso en visualización y comparación
    con imágenes RGB reales (JPG).

    Args:
        path (str): Ruta al archivo TIF multibanda ya coregistrado.

    Returns:
        tuple:
            - red (np.ndarray): Banda correspondiente al rojo (Red).
            - green (np.ndarray): Banda correspondiente al verde (Green).
            - blue (np.ndarray): Banda correspondiente al azul (Blue).

    Notas:
        - Se utiliza `reshape_as_image()` de Rasterio para reorganizar las bandas a formato (alto, ancho, bandas).
        - La imagen debe contener al menos 3 bandas. No se valida explícitamente su contenido.
        - Es importante que las bandas hayan sido coregistradas previamente para garantizar una correspondencia espacial.
    """
    with rasterio.open(path) as src:
        imagen = reshape_as_image(src.read())
        blue = imagen[:, :, 0]
        green = imagen[:, :, 1]
        red = imagen[:, :, 2]
        return red, green, blue


def normalizar_rgb_conjuntamente(r, g, b):
    """
    Normaliza simultáneamente los canales R, G y B al rango [0, 255] manteniendo su proporcionalidad relativa.

    Esta función toma como entrada los tres canales espectrales correspondientes a las bandas roja, verde y azul,
    los apila como una imagen RGB y aplica una normalización lineal conjunta. Esta normalización se realiza 
    sobre el valor mínimo y máximo global del conjunto (no por canal), asegurando una transformación uniforme
    que preserva las relaciones relativas de intensidad entre bandas.

    El resultado es una imagen RGB en formato `uint8`, adecuada para visualización, comparación con imágenes
    fotográficas, o transferencia de color.

    Args:
        r (np.ndarray): Banda del canal rojo (Red).
        g (np.ndarray): Banda del canal verde (Green).
        b (np.ndarray): Banda del canal azul (Blue).

    Returns:
        np.ndarray: Imagen RGB normalizada, con valores en el rango [0, 255] y tipo de dato `uint8`.

    Notas:
        - Se utiliza una constante pequeña (1e-5) en el denominador para evitar divisiones por cero.
        - Esta normalización conjunta es preferible cuando se desea mantener la coherencia del contraste entre bandas.
        - El orden de apilamiento es (R, G, B), compatible con librerías de visualización estándar como PIL o matplotlib.
    """
    stacked = np.dstack([r, g, b])
    min_val, max_val = stacked.min(), stacked.max()
    norm = ((stacked - min_val) / (max_val - min_val + 1e-5)) * 255
    return norm.astype(np.uint8)




# ================== FUNCIÓN DE TRANSFERENCIA DE COLOR (Reinhard et al.) ==================
def transfer_color_reinhard(source_rgb, target_rgb):
    """
    Aplica transferencia de color del estilo Reinhard para ajustar la apariencia de una imagen RGB a otra.

    Esta técnica adapta la imagen `source_rgb` (por ejemplo, generada desde bandas multiespectrales) a la apariencia
    visual de la imagen `target_rgb` (por ejemplo, una fotografía JPG), igualando sus estadísticas de color en el
    espacio de color perceptual CIELAB (L\*a\*b\*). El resultado es una imagen RGB con aspecto más natural y coherente
    con la percepción humana del color.

    Procedimiento:
    1. Ambas imágenes se convierten del espacio RGB a LAB.
    2. Se calculan la media y desviación estándar de cada canal (L, a, b) en ambas imágenes.
    3. Se ajusta la imagen fuente escalando su media y desviación para igualar las de la imagen objetivo.
    4. Se convierte la imagen corregida de regreso a RGB.

    Args:
        source_rgb (np.ndarray): Imagen RGB fuente, cuyos colores se desean adaptar (por ejemplo, RGB desde TIF).
        target_rgb (np.ndarray): Imagen RGB de referencia, cuyo estilo de color se desea imitar (por ejemplo, JPG original).

    Returns:
        np.ndarray: Imagen RGB resultante con transferencia de color aplicada, en formato `uint8`.

    Notas:
        - La transformación se realiza canal por canal (L, a, b), asumiendo que las imágenes están bien alineadas espacialmente.
        - Esta implementación sigue el enfoque de Reinhard et al. (2001), ampliamente usado en síntesis de color y análisis perceptual.
        - La imagen resultante es adecuada para tareas como segmentación, visualización o generación de máscaras sobre RGB naturalizadas.

    Referencia:
        Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001). 
        "Color Transfer between Images." IEEE Computer Graphics and Applications, 21(5), 34–41.
    """
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = cv2.meanStdDev(source_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
    transferred = (source_lab - src_mean.T) / (src_std.T + 1e-5)
    transferred = transferred * tgt_std.T + tgt_mean.T
    transferred = np.clip(transferred, 0, 255).astype(np.uint8)
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)




# ================== FUNCIONES DE PROCESAMIENTO EN LOTE (POR DIRECTORIOS) ==================
def generar_rgb_transferido_batch(directorio_tif, directorio_jpg, directorio_salida, verbose=True):
    """
    Genera imágenes RGB con transferencia de color desde imágenes multiespectrales coregistradas a imágenes RGB fotográficas.

    Esta función produce una imagen RGB visualmente natural a partir de las bandas 3, 2 y 1 (Red, Green, Blue)
    extraídas desde archivos TIF multibanda previamente coregistrados. Para lograr una apariencia coherente con
    la percepción humana del color, se realiza una transferencia de estilo de color utilizando como referencia
    visual la imagen RGB original (JPG) geométricamente corregida.

    Flujo de procesamiento por imagen:
        1. Se identifica la pareja TIF + JPG con nombre base común.
        2. Se extraen y normalizan las bandas RGB desde el TIF.
        3. Se carga y adapta el JPG (cambio de espacio de color, redimensionamiento).
        4. Se aplica transferencia de color mediante el método de Reinhard.
        5. Se guarda la imagen RGB generada con sufijo `_rgb_transferido.jpg`.

    Args:
        directorio_tif (str): Ruta al directorio que contiene archivos TIF multibanda coregistrados.
        directorio_jpg (str): Ruta al directorio que contiene imágenes RGB (.JPG) corregidas geométricamente.
        directorio_salida (str): Ruta del directorio donde se guardarán las imágenes RGB transferidas.
        verbose (bool, optional): Si es True (por defecto), imprime información del progreso por imagen.

    Notas:
        - Las imágenes TIF deben tener al menos 3 bandas y estar alineadas geométricamente.
        - Las JPG deben estar disponibles con el mismo nombre base (sin extensión) que el TIF.
        - La imagen generada está en formato `.jpg` con canales RGB tipo `uint8`, adecuada para tareas de segmentación,
          inspección visual o como entrada a modelos que requieren imágenes naturales.
    """
    os.makedirs(directorio_salida, exist_ok=True)

    for nombre_tif in os.listdir(directorio_tif):
        if not nombre_tif.lower().endswith(".tif"):
            continue

        nombre_base = os.path.splitext(nombre_tif)[0]
        ruta_tif = os.path.join(directorio_tif, nombre_tif)

        ruta_jpg = os.path.join(directorio_jpg, f"{nombre_base}.JPG")
        if not os.path.exists(ruta_jpg):
            if verbose:
                print(f"No se encontró JPG para {nombre_base}")
            continue

        try:
            if verbose:
                print(f"\nProcesando: {nombre_base}")

            r, g, b = extraer_rgb_desde_tif(ruta_tif)
            rgb_tif = normalizar_rgb_conjuntamente(r, g, b)

            rgb_jpg = cv2.imread(ruta_jpg)
            if rgb_jpg is None:
                print(f"No se pudo cargar el JPG: {ruta_jpg}")
                continue
            rgb_jpg = cv2.cvtColor(rgb_jpg, cv2.COLOR_BGR2RGB)

            if rgb_jpg.shape[:2] != rgb_tif.shape[:2]:
                rgb_jpg = cv2.resize(rgb_jpg, (rgb_tif.shape[1], rgb_tif.shape[0]), interpolation=cv2.INTER_AREA)

            rgb_transferido = transfer_color_reinhard(rgb_tif, rgb_jpg)

            salida_path = os.path.join(directorio_salida, f"{nombre_base}.jpg")
            Image.fromarray(rgb_transferido).save(salida_path)

            if verbose:
                print(f"Imagen guardada: {salida_path}")

        except Exception as e:
            print(f"Error procesando {nombre_base}: {e}")