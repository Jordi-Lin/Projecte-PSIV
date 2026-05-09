import cv2
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# ---------------- CONFIGURACIÓ ----------------
# Fitxer JSON amb rutes relatives d'imatges a processar
dataset = int(input("1: IIIT5K o 2: EMNIST: "))
if dataset == 1:
    PATH = os.path.join(os.getcwd(), 'Datasets', 'IIIT5K', 'train')
    JSON_PATH = os.path.join(os.getcwd(), 'IIIT5K.json')
elif dataset == 2:
    PATH = os.path.join(os.getcwd(), 'Datasets', 'EMNIST', 'train')
    JSON_PATH = os.path.join(os.getcwd(), 'EMNIST.json')
# Directori on es guardaran les imatges de sortida
OUTPUT_DIR = os.path.join(os.getcwd(), 'resultats_segmentacio')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MATRICES_DIR = os.path.join(OUTPUT_DIR, 'matrius')
os.makedirs(MATRICES_DIR, exist_ok=True)

# Carreguem les rutes de les imatges a processar des del JSON
try:
    with open(JSON_PATH, 'r') as f:
        paths = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {JSON_PATH}")
    paths = []

dividir = dataset == 1  # Només dividim en caràcters per IIIT5K, ja que EMNIST ja són caràcters individuals


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
    min_height = max(3, int(0.12 * H))
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
        return None, None, original_rgb.copy(), None

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, W)
    y2 = min(y + h + padding, H)

    crop_rgb = original_rgb[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    boxed = original_rgb.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return crop_rgb, crop_mask, boxed, (x1, y1, x2, y2)


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


def segment_letters_as_canny_matrices(word_crop_rgb, edges_crop, edges_overlay, output_dir, matrices_dir, image_index, output_size=32, padding=3):
    """
    Segmenta lletres utilitzant només els píxels detectats per Canny.

    Retorna:
    - letter_matrices: matrius binàries 32x32, una per lletra
    - letter_boxes: bounding boxes de cada lletra
    - debug_img: imatge amb caixes dibuixades
    - connected_edges: vores connectades
    """

    H, W = edges_crop.shape

    # --- 1. Convertir Canny a binari ---
    canny_bin = (edges_crop > 0).astype(np.uint8) * 255

    # Eliminar vores externes del crop
    border = max(1, W // 100)
    canny_bin[:, :border] = 0
    canny_bin[:, -border:] = 0
    canny_bin[:border, :] = 0
    canny_bin[-border:, :] = 0

    # --- 2. Connectar fragments propers de la mateixa lletra ---
    # Kernel en creu: connecta parts properes sense unir tant horitzontalment
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    connected_edges = cv2.dilate(canny_bin, kernel, iterations=1)

    connected_edges = cv2.morphologyEx(connected_edges, cv2.MORPH_CLOSE, kernel)

    # --- 3. Components connectats sobre Canny ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected_edges, connectivity=8)

    components = []

    min_area = max(3, int(0.0001 * H * W))  # Àrea mínima depenent de la mida del crop
    min_height = max(3, int(0.08 * H))  # Altura mínima depenent de la mida del crop

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if h < min_height:
            continue

        # Evitar línies horitzontals molt llargues
        if w > 0.85 * W and h < 0.25 * H:
            continue

        components.append({"label": i, "box": [x, y, x + w, y + h], "area": area})

    components = sorted(components, key=lambda c: c["box"][0])

    # --- 4. Agrupar fragments que pertanyen a la mateixa lletra ---
    groups = []

    def x_overlap_ratio(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        min_width = min(ax2 - ax1, bx2 - bx1)

        if min_width <= 0:
            return 0

        return overlap / float(min_width)

    def union_box(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        return [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]

    for comp in components:
        comp_box = comp["box"]
        assigned = False

        for group in groups:
            group_box = group["box"]

            overlap_x = x_overlap_ratio(comp_box, group_box)

            comp_cx = (comp_box[0] + comp_box[2]) / 2
            group_cx = (group_box[0] + group_box[2]) / 2

            group_width = group_box[2] - group_box[0]

            same_letter_by_overlap = overlap_x > 0.25
            same_letter_by_center = abs(comp_cx - group_cx) < 0.35 * max(group_width, 1)

            if same_letter_by_overlap or same_letter_by_center:
                group["labels"].append(comp["label"])
                group["box"] = union_box(group_box, comp_box)
                assigned = True
                break

        if not assigned:
            groups.append({"labels": [comp["label"]], "box": comp_box})

    # Ordenar grups d'esquerra a dreta
    groups = sorted(groups, key=lambda g: g["box"][0])

    # --- 5. Construir una matriu per cada lletra ---
    letter_matrices = []
    letter_boxes = []

    debug_img = word_crop_rgb.copy()

    for idx, group in enumerate(groups):
        group_mask = np.zeros_like(edges_crop)

        for label_id in group["labels"]:
            group_mask[labels == label_id] = 255

        coords = cv2.findNonZero(group_mask)

        if coords is None:
            continue

        x, y, w, h = cv2.boundingRect(coords)

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, W)
        y2 = min(y + h + padding, H)

        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        # Matriu binària de la lletra
        letter_mask = group_mask[y1:y2, x1:x2]

        letter_matrix = cv2.resize(letter_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

        letter_matrix = (letter_matrix > 0).astype(np.uint8)

        letter_matrices.append(letter_matrix)
        letter_boxes.append((x1, y1, x2, y2))

        # Crop amb Canny pintat
        letter_overlay = edges_overlay[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(output_dir, f'img_{image_index}_letter_{idx}_canny.png'), cv2.cvtColor(letter_overlay, cv2.COLOR_RGB2BGR))

        # Guardar la matriu com imatge
        cv2.imwrite(os.path.join(MATRICES_DIR, f'img_{image_index}_letter_{idx}_matrix.png'), letter_matrix * 255)

        # Guardar la matriu real com .npy
        np.save(os.path.join(MATRICES_DIR, f'img_{image_index}_letter_{idx}_matrix.npy'), letter_matrix)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return letter_matrices, letter_boxes, debug_img, connected_edges


def save_single_character_canny_matrix(edges_crop, matrices_dir, character, image_index, output_size=32):
    letter_matrix = cv2.resize(
        edges_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST
    )

    letter_matrix = (letter_matrix > 0).astype(np.uint8)

    filename_base = f'{character}_img_{image_index}_matrix'

    cv2.imwrite(
        os.path.join(matrices_dir, f'{filename_base}.png'),
        letter_matrix * 255
    )

    np.save(
        os.path.join(matrices_dir, f'{filename_base}.npy'),
        letter_matrix
    )

    return letter_matrix


# ---------------- PIPELINE PRINCIPAL ----------------

#for j in range(min(5, len(paths))):
total_images = 0

for root, dirs, files in os.walk(PATH):
    image_files = [
        f for f in files
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    total_images += len(image_files)

print(f"Total d'imatges: {total_images}")

for j in range(total_images):
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
        if dataset == 1:
            # Retallem les parts que no ens interessen en el cas de IIIT5K
            word_crop, mask_crop, img_box, bbox = get_word_crop(img_rgb_big, img_final, padding=5)  # Utilitzem padding=5 per assegurar-nos que no retallem les vores de les lletres
        else:
            # Per EMNIST, assumim que cada imatge ja és un sol caràcter, així que no fem crop
            word_crop = None
            mask_crop = None
            img_box = img_rgb_big.copy()
            bbox = None

        # Comprovem si s'ha obtingut un crop vàlid; si no, utilitzem la imatge completa per a les següents etapes
        if word_crop is None:
            crop_to_show = img_rgb_big
            crop_gray = img_gray
        else:
            crop_to_show = word_crop
            crop_gray = cv2.cvtColor(word_crop, cv2.COLOR_RGB2GRAY)

        # --- 8. CANNY SOBRE EL CROP ORIGINAL ---
        edges_crop = get_canny_edges(crop_gray)

        # Guardar matriu Canny en el cas EMNIST
        if dataset == 2:
            character = os.path.basename(os.path.dirname(paths[j]))
            save_single_character_canny_matrix(edges_crop, MATRICES_DIR, character, j)

        # --- 9. OVERLAY CANNY SOBRE CROP ---
        if word_crop is not None:
            edges_overlay = overlay_edges_on_crop(word_crop, edges_crop)
        else:
            edges_overlay = overlay_edges_on_crop(img_rgb_big, edges_crop)
        
        # --- 10. SEGMENTACIÓ EN CARÀCTERS AMB PROJECCIÓ VERTICAL ---
        if dividir:
            v_proj = None
            gaps = []
            char_images = []
            char_boxes = []
            edges_connected = None
            debug_canny_components = None
            letter_matrices = []
            letter_boxes = []
            debug_letters = None
            connected_edges = None

            if word_crop is not None:
                letter_matrices, letter_boxes, debug_letters, connected_edges = segment_letters_as_canny_matrices(word_crop, edges_crop, edges_overlay, OUTPUT_DIR, MATRICES_DIR, j)


        # --- Guardem el resultat i visualitzem ---
        plt.figure(figsize=(16, 8))

        if dividir:
            if debug_letters is not None:
                divisio_caracters = debug_letters
            else:
                raise ValueError("No s'ha pogut generar la imatge de divisió per caràcters (debug_letters és None)")

        images = [img_rgb_big, img_final, edges_overlay]
        titles = ['Original escalada', 'Màscara final', 'Canny + Crop']

        if dividir:
            images.append(divisio_caracters)
            titles.append('Divisió per caràcters')

        for i in range(len(images)):
            plt.subplot(2, 2, i + 1)

            if i == 1:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])

            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'Resultats_imatge{j}.png'), dpi=200)
        plt.close('all')

        print(f"Imatge {j} processada")

    except Exception as e:
        print(f"Error en {j}: {e}")
