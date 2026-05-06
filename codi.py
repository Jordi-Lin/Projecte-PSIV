import cv2
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# ---------------- CONFIGURACIÓ ----------------
# Ruta de les imatges d'entrenament a la subcarpeta Datasets/IIIT5K/train
PATH = os.path.join(os.getcwd(), 'Datasets', 'IIIT5K', 'train')
# Fitxer JSON amb rutes relatives d'imatges a processar
JSON_PATH = os.path.join(os.getcwd(), 'paths_imagenes_faciles.json')
# Directori on es guardaran les imatges de sortida
OUTPUT_DIR = os.path.join(os.getcwd(), 'resultats_segmentacio')

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(JSON_PATH, 'r') as f:
        paths = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {JSON_PATH}")
    paths = []


# ---------------- FUNCIONS AUXILIARS ----------------

def get_canny_edges(gray_img):
    """
    Detecta les vores usant l'operador Canny amb llindars adaptatius.

    Procediment:
    - Aplica un desfocalitzat Gaussià suau per reduir soroll.
    - Calcula la mediana de la imatge per derivar llindars inferiors i superiors (estratègia heurística basada en la mediana de la imatge).
    - Executa `cv2.Canny` amb aquests llindars.

    Args:
        gray_img (np.ndarray): imatge en escala de grisos (uint8).

    Retorna:
        np.ndarray: mapa binari de vores (0 o 255).
    """
    blur = cv2.GaussianBlur(gray_img, (3, 3), 0)

    median = np.median(blur)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    edges = cv2.Canny(blur, lower, upper)

    return edges


def remove_horizontal_lines(binary_img):
    """
    Detecta i elimina línies horitzontals llargues de la imatge binaritzada.

    Notes:
    - La funció assumeix text en blanc sobre fons negre (255 = foreground).
    - S'utilitza una operació de morfologia (`MORPH_OPEN`) amb un element estructurant ample i pla (1 píxel d'altura) per extreure línies horitzontals persistents.

    Args:
        binary_img (np.ndarray): imatge binària (0/255) amb text i soroll.

    Retorna:
        tuple: (img_without_lines, detected_lines)
            - img_without_lines: imatge binària amb les línies restades.
            - detected_lines: imatge binària amb només les línies detectades.
    """
    _, W = binary_img.shape

    # Amplada del kernel proporcional a l'amplada de la imatge, amb un mínim
    horizontal_kernel_width = max(15, W // 3)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1)
    )

    detected_lines = cv2.morphologyEx(
        binary_img,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    img_without_lines = cv2.subtract(binary_img, detected_lines)

    return img_without_lines, detected_lines


def filter_text_components(binary_img):
    """
    Filtra components connectats deixant només aquells que semblen lletres o parts de paraules.

    Estratègia:
    - Calcula components connectats amb `connectedComponentsWithStats`.
    - Aplica regles heurístiques (àrea mínima, altura mínima/màxima, aspect ratio) per eliminar soroll, regles i formes no textuals.

    Args:
        binary_img (np.ndarray): imatge binària (0/255).

    Retorna:
        np.ndarray: imatge binària filtrada amb només components acceptats (255).
    """
    H, W = binary_img.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    filtered = np.zeros_like(binary_img)

    # Llindars dependents de la mida de la imatge
    min_area = max(4, int(0.0002 * H * W))
    min_height = max(3, int(0.10 * H))
    max_height = int(0.98 * H)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        aspect_ratio = w / float(h) if h > 0 else 0

        # Regles per identificar formes que NO volem conservar
        is_horizontal_line = w > 0.60 * W and h < 0.20 * H
        is_tiny_noise = area < min_area
        is_too_short = h < min_height
        is_too_tall = h > max_height
        is_weird_shape = aspect_ratio > 8 and h < 0.35 * H

        if not (
            is_horizontal_line
            or is_tiny_noise
            or is_too_short
            or is_too_tall
            or is_weird_shape
        ):
            # Manté el component si no compleix cap condició d'eliminació
            filtered[labels == i] = 255

    return filtered


def get_word_crop(original_rgb, mask, padding=5):
    """
    Obté el rectangle que engloba la màscara (assumint que representa una paraula)
    i retorna el crop corresponent sobre la imatge RGB original.

    Args:
        original_rgb (np.ndarray): imatge RGB original (H x W x 3).
        mask (np.ndarray): imatge binària amb la màscara de la paraula (H x W).
        padding (int): píxels addicionals al voltant del bounding box.

    Retorna:
        tuple: (crop, boxed, bbox)
            - crop: subimatge corresponent al bounding box (o None si no hi ha coords).
            - boxed: imatge RGB original amb el bounding box dibuixat (per visualització).
            - bbox: tupla (x1, y1, x2, y2) amb coordenades del rectangle o None.
    """
    H, W = mask.shape
    coords = cv2.findNonZero(mask)

    if coords is None:
        return None, original_rgb.copy(), None

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, W)
    y2 = min(y + h + padding, H)

    crop = original_rgb[y1:y2, x1:x2]

    boxed = original_rgb.copy()
    cv2.rectangle(
        boxed,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        1
    )

    return crop, boxed, (x1, y1, x2, y2)


def overlay_edges_on_crop(crop_rgb, edges):
    """
    Dibuixa (pinta) les vores detectades sobre la imatge crop com a píxels vermells.

    Args:
        crop_rgb (np.ndarray): crop RGB sobre el qual superposar les vores.
        edges (np.ndarray): mapa binari de vores (0/255) amb la mateixa mida que `crop_rgb`.

    Retorna:
        np.ndarray: imatge RGB amb les vores pintades en vermell.
    """
    overlay = crop_rgb.copy()

    overlay[edges > 0] = [255, 0, 0]

    return overlay


# ---------------- PIPELINE PRINCIPAL ----------------

for j in range(min(10, len(paths))):
    image_path = os.path.join(PATH, paths[j])

    try:
        # Carrega la imatge en BGR (format OpenCV)
        img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            print(f"No s'ha pogut llegir la imatge: {image_path}")
            continue

        # --- 1. LECTURA I ESCALAT ---
        # Convertim a RGB per fer visualitzacions coherents amb matplotlib
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Escalem la imatge per treballar amb més resolució (millor separació de caràcters)
        scale = 3  # Utilitzem 3 com factor, ja que les imatges d'entrada són petites (100-200px d'amplada)
        img_rgb_big = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        img_gray = cv2.cvtColor(img_rgb_big, cv2.COLOR_RGB2GRAY)

        # --- 2. PREPROCESSAMENT ---
        # Utilitzem CLAHE per millorar el contrast local i fer que les lletres siguin més destacades respecte al fons
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # Suau difuminat per reduir soroll abans de la binarització
        img_blur = cv2.GaussianBlur(img_clahe, (3, 3), 0)

        # --- 3. BINARITZACIÓ PER OBTENIR MÀSCARA ---
        # Otsu automàticament tria el llindar; THRESH_BINARY_INV posa el text a 255
        _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- 4. ELIMINAR LÍNIES HORITZONTALS ---
        img_no_lines, detected_lines = remove_horizontal_lines(img_bin)

        # --- 5. MORFOLOGIA SUAU ---
        # Utilitzem la funció cv2.morphologyEx amb cv2.MORPH_CLOSE per tanca petits forats i unir fragments de caràcters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Kernel petit per no unir paraules diferents
        img_morph = cv2.morphologyEx(img_no_lines, cv2.MORPH_CLOSE, kernel_close)

        # --- 6. FILTRAT DE COMPONENTS ---
        img_filtered = filter_text_components(img_morph)

        # Guardem una copia per visualització posterior
        img_final = img_filtered.copy()

        # --- 7. CROP DE PARAULA ---
        word_crop, img_box, bbox = get_word_crop(img_rgb_big, img_final, padding=5)

        # Comprovem si s'ha obtingut un crop vàlid; si no, utilitzem la imatge completa per a les següents etapes
        if word_crop is None:
            crop_to_show = img_rgb_big
            crop_gray = img_gray
        else:
            crop_to_show = word_crop
            crop_gray = cv2.cvtColor(word_crop, cv2.COLOR_RGB2GRAY)

        # --- 8. CANNY SOBRE EL CROP ORIGINAL ---
        edges_crop = get_canny_edges(crop_gray)

        # --- 9. OVERLAY CANNY SOBRE CROP ---
        if word_crop:
            edges_overlay = overlay_edges_on_crop(word_crop, edges_crop)
        else:
            edges_overlay = overlay_edges_on_crop(img_rgb_big, edges_crop)


        # --- Guardem el resultat i visualitzem ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'mascara_final_{j}.png'), img_final)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'canny_lletres_{j}.png'), edges_crop)

        if word_crop is not None:
            word_crop_bgr = cv2.cvtColor(word_crop, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(edges_overlay, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(OUTPUT_DIR, f'crop_paraula_{j}.png'), word_crop_bgr)

            cv2.imwrite(os.path.join(OUTPUT_DIR, f'vores_sobre_crop_{j}.png'), overlay_bgr)

        plt.figure(figsize=(18, 10))

        titles = ['Original escalada', 'CLAHE + GaussianBlur', 'Otsu invertit', 'Línies detectades', 'Sense línies', 'Màscara final', 'Crop paraula', 'Canny vores', 'Vores sobre crop']
        images = [img_rgb_big, img_blur, img_bin, detected_lines, img_no_lines, img_final, crop_to_show, edges_crop, edges_overlay]

        for i in range(9):
            plt.subplot(3, 3, i + 1)

            if i in [0, 6, 8]:
                plt.imshow(images[i])
            else:
                plt.imshow(images[i], cmap='gray')

            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'comparativa_canny_{j}.png'), dpi=200)
        plt.close('all')

        print(f"Processada imatge {j}: {paths[j]}")
        print(f"Bounding box: {bbox}")

    except Exception as e:
        print(f"Error en {j}: {e}")