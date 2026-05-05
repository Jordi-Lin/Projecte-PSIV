import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# Configuración de rutas
PATH = r"D:\Biblioteca\Desktop\PISV\Projecte\Datasets\IIIT5K\train"
JSON_PATH = 'paths_imagenes_faciles.json'

try:
    with open(JSON_PATH, 'r') as f:
        paths = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {JSON_PATH}")
    paths = []

for j in range(min(10, len(paths))):
    IMAGE_NAME = os.path.join(PATH, paths[j])

    try:
        img_bgr = cv2.imread(IMAGE_NAME)
        if img_bgr is None: continue

        # --- 1. PREPROCESAMIENTO "CHETADO" ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # CLAHE + Bilateral (suavizado sin perder bordes)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_pre = clahe.apply(img_gray)
        img_blur = cv2.bilateralFilter(img_pre, 9, 75, 75)

        # --- 2. BINARIZACIÓN Y MORFOLOGÍA ---
        img_bin = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 15, 8)
        
        # Opening para quitar ruido pequeño
        kernel = np.ones((2, 2), np.uint8)
        img_open = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

        # --- 3. FILTRADO POR COMPONENTES CONECTADOS ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_open, connectivity=8)
        img_final = np.zeros_like(img_open)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = w / float(h)

            # Filtramos por área mínima y descartamos líneas horizontales muy largas (bordes)
            if area > 40 and 0.1 < aspect_ratio < 4.0:
                img_final[labels == i] = 255

        # --- 4. PROYECCIONES (Detección de filas y columnas) ---
        # Invertimos para que el texto sume (texto blanco sobre fondo negro)
        # Si ya es blanco sobre negro, no inviertas. 
        # Aquí asumimos que img_final tiene texto en 255.
        h_proj = np.sum(img_final, axis=1)
        v_proj = np.sum(img_final, axis=0)

        # --- 5. VISUALIZACIÓN ---
        # Gráfica de comparativa principal
        plt.figure(figsize=(18, 10))
        titles = ['Original RGB', 'CLAHE + Bilateral', 'Otsu (Referencia)', 
                  'Adaptativa Base', 'Canny de Final', 'RESULTADO FINAL']
        
        # Generamos un Canny solo para comparar en la tabla
        edges = cv2.Canny(img_final, 100, 200)
        
        _, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_adapt_simple = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        images = [img_rgb, img_blur, img_otsu, img_adapt_simple, edges, img_final]

        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray' if i > 0 else None)
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'comparativa_{j}.png', dpi=200)
        plt.close()

        # Gráfica de Proyecciones
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(h_proj)
        plt.title('Proyección Horizontal (Filas)')
        plt.subplot(1, 2, 2)
        plt.plot(v_proj)
        plt.title('Proyección Vertical (Letras)')
        plt.savefig(f'proyecciones_{j}.png')
        plt.close()

        print(f"Procesada imagen {j}")

    except Exception as e:
        print(f"Error en {j}: {e}")