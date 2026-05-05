import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PATH = r"D:\Biblioteca\Desktop\PISV\Projecte\Datasets"
IMAGE_NAME = os.path.join(PATH, "escola00.jpeg")

try:
    img_bgr = cv2.imread(IMAGE_NAME)
    
    if img_bgr is None:
        raise Exception(f"No s'ha pogut carregar l'imatge a: {IMAGE_NAME}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


    _, img_bin_simple = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    _, img_bin_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['Original RGB', 'Escala de Grises', 'Binaria (Fija 127)', 'Binaria (Otsu)', 'Binaria (Adaptativa)']
    images = [img_rgb, img_gray, img_bin_simple, img_bin_otsu, img_bin_adapt]

    plt.figure(figsize=(15, 10))

    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('comparativa_binarizacion.png', dpi=200)
    print("Procesamiento completado. Imagen 'comparativa_binarizacion.png' guardada.")

except Exception as e:
    print(f"Error detectado: {e}")