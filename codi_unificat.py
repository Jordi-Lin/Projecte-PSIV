import cv2
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from scipy.io import loadmat

# ---------------- CONFIGURACIÓ ----------------
# Fitxer JSON amb rutes relatives d'imatges a processar
dataset = int(input("1: IIIT5K o 2: EMNIST: "))
if dataset == 1:
    PATH = os.path.join(os.getcwd(), 'Datasets', 'IIIT5K', 'train')
    JSON_PATH = os.path.join(os.getcwd(), 'IIIT5K.json')

    # Fitxer MAT amb labels
    MAT_PATH = os.path.join(os.getcwd(), 'Datasets', 'IIIT5K', 'traindata.mat')

    # Carregar dades MATLAB
    mat = loadmat(MAT_PATH, squeeze_me=True)

    # Contingut principal
    iiit_data = mat['traindata']

    # Diccionari: nom imatge -> etiqueta
    iiit_labels = {}

    for item in iiit_data:
        image_name = str(item[0])
        label = str(item[1])

        # Guardem només el nom del fitxer
        image_name = os.path.basename(image_name)

        iiit_labels[image_name] = label
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
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))

    detected_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)

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

    # Recorrem els components detectats i apliquem filtres per eliminar soroll i fragments no textuals
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
        """Calcula la proporció d'solapament horitzontal entre dues caixes, normalitzada pel mínim ample de les dues caixes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        min_width = min(ax2 - ax1, bx2 - bx1)

        if min_width <= 0:
            return 0

        return overlap / float(min_width)

    def union_box(box_a, box_b):
        """Retorna la caixa que engloba completament les dues caixes donades."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        return [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]

    # Recorrem els components i els agrupem segons la proximitat horitzontal i la superposició
    for comp in components:
        comp_box = comp["box"]
        assigned = False

        for group in groups:
            group_box = group["box"]

            overlap_x = x_overlap_ratio(comp_box, group_box)

            # Calcul de centres horitzontals per a una regla addicional de proximitat
            comp_cx = (comp_box[0] + comp_box[2]) / 2
            group_cx = (group_box[0] + group_box[2]) / 2

            # Ample de la caixa del grup per a la regla de proximitat basada en centres
            group_width = group_box[2] - group_box[0]

            # Considerem que pertanyen a la mateixa lletra si tenen una superposició horitzontal significativa o si els centres estan prou propers (per evitar separar fragments que no s'han connectat però que clarament són de la mateixa lletra)
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
    """Funció específica per guardar la matriu Canny d'una imatge EMNIST, on cada imatge ja és un caràcter individual."""
    letter_matrix = cv2.resize(edges_crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    letter_matrix = (letter_matrix > 0).astype(np.uint8)

    filename_base = f'{character}_img_{image_index}_matrix'

    np.save(os.path.join(matrices_dir, f'{filename_base}.npy'), letter_matrix)

    return letter_matrix

def segment_letters_by_labels_binary(word_crop_rgb, mask_crop, output_dir, matrices_dir, image_index, output_size=32, padding=3, min_area=200, min_width=5, min_height=10, max_aspect_ratio=3.0, min_aspect_ratio=0.1):
    """
    Segmenta lletres a partir de labels/components connectats, adaptant el mètode del codi 2.

    En comptes de només dibuixar rectangles com al codi 2, aquesta versió:
    - treballa sobre la màscara binària ja retallada de la paraula,
    - aplica una petita neteja morfològica,
    - calcula components connectats / labels,
    - filtra regions per àrea, mida i aspect ratio,
    - guarda una matriu 32x32 per cada component detectat,
    - retorna una imatge debug amb les caixes dibuixades.
    """

    H, W = mask_crop.shape

    # Convertir a binària 0/255
    binary = (mask_crop > 0).astype(np.uint8) * 255

    # Neteja similar a la del codi 2, però amb OpenCV per evitar dependències extra
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1
    )

    # Labels / components connectats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []

    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if w < min_width or h < min_height:
            continue

        aspect_ratio = w / float(h) if h > 0 else 0

        if aspect_ratio > max_aspect_ratio or aspect_ratio < min_aspect_ratio:
            continue

        components.append({
            "label": label_id,
            "box": (x, y, x + w, y + h),
            "area": area
        })

    # Ordenar d'esquerra a dreta
    components = sorted(components, key=lambda c: c["box"][0])

    letter_matrices = []
    letter_boxes = []
    debug_img = word_crop_rgb.copy()

    for idx, component in enumerate(components):
        label_id = component["label"]
        x1, y1, x2, y2 = component["box"]

        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, W)
        y2 = min(y2 + padding, H)

        if x2 - x1 < 4 or y2 - y1 < 6:
            continue

        # Guardem només el component d'aquest label, no tota la finestra
        component_mask = np.zeros_like(binary)
        component_mask[labels == label_id] = 255

        letter_mask = component_mask[y1:y2, x1:x2]

        letter_matrix = cv2.resize(
            letter_mask,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST
        )

        letter_matrix = (letter_matrix > 0).astype(np.uint8)

        letter_matrices.append(letter_matrix)
        letter_boxes.append((x1, y1, x2, y2))

        cv2.imwrite(
            os.path.join(output_dir, f'img_{image_index}_letter_{idx}_labels.png'),
            letter_mask
        )

        cv2.imwrite(
            os.path.join(matrices_dir, f'img_{image_index}_letter_{idx}_matrix.png'),
            letter_matrix * 255
        )

        np.save(
            os.path.join(matrices_dir, f'img_{image_index}_letter_{idx}_matrix.npy'),
            letter_matrix
        )

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return letter_matrices, letter_boxes, debug_img, labels


def smooth_1d(signal, kernel_size):
    """Suavitza un vector 1D amb una mitjana mòbil."""
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return signal.astype(np.float32)

    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal.astype(np.float32), kernel, mode='same')


def merge_close_intervals(intervals, max_gap):
    """Uneix intervals [x1, x2] que estan separats per pocs píxels."""
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda t: t[0])
    merged = [list(intervals[0])]

    for x1, x2 in intervals[1:]:
        prev_x1, prev_x2 = merged[-1]

        if x1 - prev_x2 <= max_gap:
            merged[-1][1] = max(prev_x2, x2)
        else:
            merged.append([x1, x2])

    return [(x1, x2) for x1, x2 in merged]


def split_wide_intervals(intervals, x_projection, expected_width):
    """
    Si una finestra és massa ampla, intenta dividir-la buscant mínims locals
    de la projecció vertical. Això ajuda quan dues lletres estan enganxades.
    """
    if expected_width <= 0:
        return intervals

    result = []

    for x1, x2 in intervals:
        width = x2 - x1

        if width <= 1.8 * expected_width:
            result.append((x1, x2))
            continue

        segment = x_projection[x1:x2]
        n_splits = int(round(width / expected_width))

        if n_splits <= 1:
            result.append((x1, x2))
            continue

        cut_points = []

        for k in range(1, n_splits):
            approx = int(k * width / n_splits)
            search_radius = max(2, int(0.20 * expected_width))
            left = max(0, approx - search_radius)
            right = min(len(segment), approx + search_radius + 1)

            if right <= left:
                continue

            local_min = left + int(np.argmin(segment[left:right]))
            cut_points.append(x1 + local_min)

        prev = x1
        for cut in sorted(set(cut_points)):
            if cut - prev >= 3:
                result.append((prev, cut))
            prev = cut

        if x2 - prev >= 3:
            result.append((prev, x2))

    return result


def segment_letters_sliding_window_binary(mask_crop, output_dir, matrices_dir, image_index, output_size=32, padding=3, smooth_window=7, empty_ratio=0.035, min_char_width_ratio=0.035, max_char_width_ratio=0.16):
    """
    Segmentació millorada amb finestra lliscant sobre la imatge binària.
    Està pensada per evitar talls interns dins de lletres com O, A, S, N.
    """

    H, W = mask_crop.shape

    binary = (mask_crop > 0).astype(np.uint8) * 255

    # 1. Eliminar línies horitzontals llargues si encara existeixen
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, W // 3), 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    binary = cv2.subtract(binary, horizontal_lines)

    # 2. Neteja suau
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    binary01 = (binary > 0).astype(np.uint8)

    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    cols = np.where(np.sum(binary01, axis=0) > 0)[0]

    if len(cols) == 0:
        return [], [], debug_img, None

    text_x1 = int(cols[0])
    text_x2 = int(cols[-1]) + 1

    text = binary01[:, text_x1:text_x2]
    Wt = text.shape[1]

    projection = np.sum(text, axis=0).astype(np.float32)

    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smooth = np.convolve(projection, kernel, mode="same")

    max_proj = max(np.max(smooth), 1)

    min_char_width = max(6, int(min_char_width_ratio * Wt))
    max_char_width = max(14, int(max_char_width_ratio * Wt))

    # 3. Buscar només columnes realment buides o gairebé buides
    low_cols = smooth <= empty_ratio * max_proj

    gaps = []
    in_gap = False
    start = 0

    for x in range(Wt):
        if low_cols[x] and not in_gap:
            start = x
            in_gap = True
        elif not low_cols[x] and in_gap:
            end = x
            in_gap = False

            if end - start >= 2:
                gaps.append((start, end))

    if in_gap:
        end = Wt
        if end - start >= 2:
            gaps.append((start, end))

    # 4. Convertir gaps en talls
    cuts = [0]

    for start, end in gaps:
        cut = (start + end) // 2

        if cut - cuts[-1] >= min_char_width:
            cuts.append(cut)

    if Wt - cuts[-1] >= min_char_width:
        cuts.append(Wt)
    else:
        cuts[-1] = Wt

    # 5. Si algun segment és massa ample, intentar dividir-lo pel millor mínim intern
    final_cuts = [cuts[0]]

    for i in range(len(cuts) - 1):
        a = cuts[i]
        b = cuts[i + 1]
        width = b - a

        if width <= max_char_width:
            final_cuts.append(b)
            continue

        # Nombre aproximat de lletres dins aquest bloc
        n_parts = max(2, int(round(width / max_char_width)))

        approx_step = width / n_parts

        for k in range(1, n_parts):
            expected = int(a + k * approx_step)

            search_radius = int(0.25 * approx_step)

            s = max(a + min_char_width, expected - search_radius)
            e = min(b - min_char_width, expected + search_radius)

            if e <= s:
                continue

            local = smooth[s:e]

            # Tall al mínim de tinta de la zona esperada
            cut = s + int(np.argmin(local))

            if cut - final_cuts[-1] >= min_char_width:
                final_cuts.append(cut)

        if b - final_cuts[-1] >= min_char_width:
            final_cuts.append(b)
        else:
            final_cuts[-1] = b

    cuts = final_cuts

    # 6. Crear caixes
    letter_matrices = []
    letter_boxes = []

    for i in range(len(cuts) - 1):
        x1_local = cuts[i]
        x2_local = cuts[i + 1]

        if x2_local - x1_local < min_char_width:
            continue

        region = text[:, x1_local:x2_local]
        coords = cv2.findNonZero((region * 255).astype(np.uint8))

        if coords is None:
            continue

        x, y, w, h = cv2.boundingRect(coords)

        lx1 = max(text_x1 + x1_local + x - padding, 0)
        ly1 = max(y - padding, 0)
        lx2 = min(text_x1 + x1_local + x + w + padding, W)
        ly2 = min(y + h + padding, H)

        if lx2 - lx1 < 4 or ly2 - ly1 < 6:
            continue

        letter_mask = binary[ly1:ly2, lx1:lx2]

        letter_matrix = cv2.resize(
            letter_mask,
            (output_size, output_size),
            interpolation=cv2.INTER_NEAREST
        )

        letter_matrix = (letter_matrix > 0).astype(np.uint8)

        letter_matrices.append(letter_matrix)
        letter_boxes.append((lx1, ly1, lx2, ly2))

        cv2.imwrite(
            os.path.join(output_dir, f'img_{image_index}_letter_{i}_sliding_binary.png'),
            letter_mask
        )

        cv2.imwrite(
            os.path.join(matrices_dir, f'img_{image_index}_letter_{i}_matrix.png'),
            letter_matrix * 255
        )

        np.save(
            os.path.join(matrices_dir, f'img_{image_index}_letter_{i}_matrix.npy'),
            letter_matrix
        )

        cv2.rectangle(debug_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

    return letter_matrices, letter_boxes, debug_img, smooth


def save_single_character_matrix(mask_crop, matrices_dir, character, image_index, output_size=32):
    """Per EMNIST: guarda directament la màscara com a matriu 32x32."""
    letter_matrix = cv2.resize(mask_crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    letter_matrix = (letter_matrix > 0).astype(np.uint8)
    filename_base = f'{character}_img_{image_index}_matrix'
    np.save(os.path.join(matrices_dir, f'{filename_base}.npy'), letter_matrix)
    cv2.imwrite(os.path.join(matrices_dir, f'{filename_base}.png'), letter_matrix * 255)
    return letter_matrix


# ---------------- PIPELINE PRINCIPAL ----------------
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

        # --- 10. SEGMENTACIÓ EN CARÀCTERS ---
        if dividir:
            letter_matrices = []
            letter_boxes = []
            debug_letters = None
            connected_edges = None

            if word_crop is not None:
                metode_segmentacio = int(input("Tria el mètode de segmentació (1: Canny + components connectats, 2: labels sobre binària, 3: projecció + finestra lliscant): "))
                if metode_segmentacio == 1:
                    # Mètode del primer fitxer: Canny + components connectats
                    letter_matrices, letter_boxes, debug_letters, connected_edges = segment_letters_as_canny_matrices(
                        word_crop, edges_crop, edges_overlay, OUTPUT_DIR, MATRICES_DIR, j
                    )
                elif metode_segmentacio == 2:
                    # Mètode del segon fitxer: labels / components connectats sobre la màscara binària
                    letter_matrices, letter_boxes, debug_letters, labels = segment_letters_by_labels_binary(
                        word_crop, mask_crop, OUTPUT_DIR, MATRICES_DIR, j,
                        min_area=200,
                        min_width=5,
                        min_height=10,
                        max_aspect_ratio=3.0,
                        min_aspect_ratio=0.1
                    )
                elif metode_segmentacio == 3:
                    # Mètode del tercer fitxer: projecció/finestra lliscant sobre la màscara binària
                    letter_matrices, letter_boxes, debug_letters, projection = segment_letters_sliding_window_binary(mask_crop, OUTPUT_DIR, MATRICES_DIR, j)
                else:
                    raise ValueError("Mètode de segmentació no vàlid. Escriu 1, 2 o 3.")


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
