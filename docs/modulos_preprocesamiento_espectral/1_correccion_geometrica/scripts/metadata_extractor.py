"""
MÓDULO DE EXTRACCIÓN DE METADATOS DE IMÁGENES MULTIESPECTRALES

Autor: Paola Andrea Mejia-Zuluaga  
Fecha: Marzo 25 de 2025  
Proyecto: Preprocesamiento de imágenes para el Proyecto - Monitoreo de especies de Muérdago  
          en Parques Urbanos Usando Imágenes Aéreas e Inteligencia Artificial  
Versión: 1.0  
Contacto: paomejia23@gmail.com  

Descripción:
Este módulo permite extraer metadatos técnicos de imágenes multiespectrales en formato `.TIF` y `.JPG`
utilizando la herramienta de línea de comandos `ExifTool`. Los metadatos se guardan en dos archivos `.csv` 
(independientes para cada tipo de imagen), y están destinados a ser utilizados en procesos posteriores de 
corrección radiométrica, análisis espectral y registro multitemporal.

El sistema identifica automáticamente todas las imágenes contenidas en una carpeta específica, extrae 
los metadatos embebidos y organiza la salida en formato tabular compatible con pandas. La estructura 
resultante facilita la integración con otros módulos de preprocesamiento y análisis.

Requisitos:
- Python 3.8 o superior
- ExifTool instalado y accesible desde línea de comandos

Entradas:
- Carpeta con imágenes multiespectrales en formato `.TIF` y/o `.JPG`

Salidas:
- `metadata_tif.csv`: Metadatos extraídos de imágenes `.TIF`
- `metadata_jpg.csv`: Metadatos extraídos de imágenes `.JPG`

Uso:
Esta función puede ser llamada directamente desde otros scripts de preprocesamiento (por ejemplo,
cuando no se encuentra un archivo de metadatos existente), o utilizada de forma independiente para
crear los archivos CSV de metadatos.

Ejemplo:
    from metadata_extractor import generate_metadata_csv
    generate_metadata_csv("ruta/a/la/carpeta_con_imagenes")
"""

# =============================== IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN ===============================
import os
import csv
import subprocess

def generate_metadata_csv(folder_path):
    """
    Extrae metadatos de archivos .TIF y .JPG usando ExifTool y los guarda en un archivo CSV.
    
    Parámetros:
    - folder_path (str): Ruta de la carpeta que contiene las imágenes.
    - output_csv (str): Ruta del archivo CSV de salida.
    
    Requiere ExifTool instalado y accesible desde línea de comandos.
    """
    # Verificar si la carpeta existe
    if not os.path.isdir(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return
    
    # Obtener todos los archivos .tif y .jpg en la carpeta
    image_files_tif = []
    image_files_jpg = []

    for root, _, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(root, f)
            if f.lower().endswith(('.tif', '.tiff')):
                image_files_tif.append(file_path)
            elif f.lower().endswith(('.jpg', '.jpeg')):
                image_files_jpg.append(file_path)

    if not image_files_tif and not image_files_jpg:
        print("No se encontraron archivos TIFF ni JPG en la carpeta.")
        return
    
    
    # Función para extraer metadatos con ExifTool
    def extract_metadata(image_files):
        """
        Extrae metadatos de una lista de imágenes utilizando ExifTool.

        Esta función recorre una lista de rutas a archivos de imagen (.TIF o .JPG) y ejecuta 
        el comando `exiftool` para cada una, extrayendo los metadatos embebidos en los encabezados 
        del archivo. La salida se transforma en un diccionario por imagen, agregando también 
        información sobre el nombre del archivo y su extensión.

        Parámetros:
        - image_files (list of str): Lista de rutas completas a archivos de imagen a procesar.

        Proceso:
        - Ejecuta `exiftool` para cada archivo.
        - Interpreta la salida línea por línea, separando clave y valor.
        - Agrega campos auxiliares como el nombre del archivo y su extensión.
        - Omite archivos que no se pueden procesar correctamente.

        Retorno:
        - list of dict: Lista de diccionarios con los metadatos extraídos de cada imagen.

        Requiere:
        - ExifTool instalado y accesible desde línea de comandos.

        Ejemplo de uso:
        metadata = extract_metadata(["ruta/imagen1.TIF", "ruta/imagen2.JPG"])
        """
        metadata_list = []
        print(f"Procesando...")
        for image_file in image_files:
            #print(f"Procesando {image_file}...")
            try:
                result = subprocess.run(['exiftool', image_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"Error al procesar {image_file}: {result.stderr}")
                    continue

                # Parsear los metadatos de la salida de ExifTool
                metadata = {}
                for line in result.stdout.splitlines():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()

                # Añadir un campo con el nombre del archivo y su tipo
                metadata['File Name'] = os.path.basename(image_file)
                metadata['File Extension'] = os.path.splitext(image_file)[1].lower()
                metadata_list.append(metadata)

            except Exception as e:
                print(f"Error procesando {image_file}: {e}")

        return metadata_list
    


    # Función para guardar metadatos en CSV
    def save_metadata(metadata_list, output_csv):
        """
        Guarda una lista de metadatos en un archivo CSV.

        Esta función toma una lista de diccionarios con metadatos extraídos de imágenes 
        (por ejemplo, usando ExifTool) y los guarda en un archivo CSV. Los encabezados 
        del archivo se determinan automáticamente a partir de todas las claves únicas 
        presentes en la lista de diccionarios.

        Parámetros:
        - metadata_list (list of dict): Lista de metadatos, donde cada diccionario 
        representa los metadatos de una imagen.
        - output_csv (str): Ruta completa del archivo CSV de salida.

        Retorno:
        - bool: True si el archivo fue guardado exitosamente. 
                None si la lista de metadatos está vacía y no se genera el archivo.

        Notas:
        - Si la lista está vacía, no se crea el archivo CSV y se imprime un mensaje.
        - La codificación del archivo es UTF-8 para asegurar compatibilidad con caracteres especiales.
        - Se utilizan todas las claves únicas presentes en los diccionarios como nombres de columna.
        """
        if not metadata_list:
            print(f"No se extrajo ningún metadato para {output_csv}.")
            return
        
        keys = sorted(set().union(*(meta.keys() for meta in metadata_list)))
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_list)
        
        print(f"Metadatos guardados en {output_csv}")
        return True 

    # Guardar archivos en la misma carpeta de origen
    output_csv_tif = os.path.join(folder_path, "metadata_tif.csv")
    output_csv_jpg = os.path.join(folder_path, "metadata_jpg.csv")

    metadata_tif = extract_metadata(image_files_tif)
    metadata_jpg = extract_metadata(image_files_jpg)

    success_tif = save_metadata(metadata_tif, output_csv_tif)
    save_metadata(metadata_jpg, output_csv_jpg)

    return success_tif