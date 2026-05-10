import os
import cv2
import json
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_canny_dataset(dataset_dir, img_size=32):
    """
    Carrega matrius .npy i utilitza el primer caràcter del nom del fitxer
    com a label.

    Exemple:
        A_img_0_matrix.npy -> label A
        0_img_3_matrix.npy -> label 0

    Paràmetres:
        dataset_dir (str): directori on es troben les matrius Canny guardades en format .npy.
        img_size (int): mida final de cada matriu. Per defecte és 32, generant imatges 32x32.

    Retorna:
        X (np.ndarray): conjunt d'imatges preprocessades amb forma (num_imatges, img_size, img_size, 1).
        y (np.ndarray): etiquetes convertides a format one-hot encoding.
        class_to_id (dict): diccionari que associa cada caràcter amb un identificador numèric.
        id_to_class (dict): diccionari invers que associa cada identificador numèric amb el caràcter corresponent.
    """

    images = []
    labels = []

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".npy"):
            continue

        path = os.path.join(dataset_dir, filename)
        img = np.load(path)

        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)  # Redimensionem la matriu a img_size x img_size. INTER_NEAREST manté valors binaris sense crear píxels intermedis.

        img = (img > 0).astype("float32")  # Convertim la matriu a binària: valors > 0 passen a 1, la resta a 0. També es convertim a float32, format adequat per TensorFlow/Keras.

        label = filename[0]  # El primer caràcter del nom del fitxer s'utilitza com a etiqueta. Exemple: 'A' per 'A_img_0_matrix.npy'

        images.append(img)
        labels.append(label)

    classes = sorted(list(set(labels)))
    class_to_id = {c: i for i, c in enumerate(classes)}  # Assignem un identificador numèric a cada classe. Exemple: {'A': 0, 'B': 1}

    id_to_class = {i: c for c, i in class_to_id.items()}  # Crea el diccionari invers per poder convertir prediccions numèriques a caràcters

    y_ids = np.array([class_to_id[label] for label in labels], dtype="int32")  # Convertim les etiquetes textuals a identificadors numèrics

    X = np.array(images, dtype="float32")  # Convertim la llista d'imatges en un array NumPy

    # Adapta la forma de les dades al format que espera una CNN:
    # (nombre_imatges, amplada, altura, canals)
    # El canal és 1 perquè les imatges són en escala de grisos/binàries
    X = X.reshape(-1, img_size, img_size, 1)

    # Converteix les etiquetes numèriques a one-hot encoding per classificació multiclasse. Exemple: classe 2 de 4 -> [0, 0, 1, 0]
    y = to_categorical(y_ids, num_classes=len(classes))

    return X, y, class_to_id, id_to_class


def build_cnn(img_size, num_classes):
    """
    Funció per construir la CNN.

    Paràmetres:
        img_size (int): mida de les imatges d'entrada. Per exemple, 32 per imatges 32x32.
        num_classes (int): nombre total de classes/caràcters que el model ha de predir.

    Retorna:
        model (Sequential): model CNN compilat i preparat per entrenar.
    """

    model = Sequential()  # Utilitzem Sequential perquè les operacions s'apliquen una rera l'altra

    # Primera capa convolucional
    # Aplica 32 filtres de mida 3x3 per detectar patrons simples com vores i formes bàsiques.
    # activation="relu" introdueix no-linealitat i ajuda el model a aprendre patrons complexos.
    # input_shape indica que l'entrada és una imatge img_size x img_size amb 1 canal.
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)))

    model.add(MaxPooling2D((2, 2)))  # Redueix la mida espacial de la imatge a la meitat. Conserva la informació més important i redueix el cost computacional

    model.add(Conv2D(64, (3, 3), activation="relu"))  # Segona capa convolucional amb 64 filtres.  Pot aprendre patrons més complexos combinant els detectats anteriorment

    model.add(MaxPooling2D((2, 2)))  # Nova reducció de dimensió per concentrar la informació més rellevant

    model.add(Conv2D(128, (3, 3), activation="relu"))  # Tercera capa convolucional amb 128 filtres. Aprèn característiques més abstractes de les formes dels caràcters

    model.add(MaxPooling2D((2, 2)))  # Reduim novament la mida de les representacions internes

    model.add(Flatten())  # Convertim la sortida 2D/3D de les convolucions en un vector 1D. Aquest vector es passarà a les capes denses.

    model.add(Dense(128, activation="relu"))  # Capa densa amb 128 neurones per combinar les característiques apreses per decidir quin caràcter representa la imatge

    model.add(Dropout(0.5))  # Durant l'entrenament desactiva aleatòriament el 50% de neurones per reduir l'overfitting i millora la generalització


    # Capa final de classificació
    # Té una neurona per cada classe
    # softmax retorna una probabilitat per cada caràcter possible
    model.add(Dense(num_classes, activation="softmax"))

    # Compilem el model
    # Adam ajusta els pesos automàticament durant l'entrenament
    # categorical_crossentropy és adequada per classificació multiclasse amb one-hot encoding
    # accuracy permet mesurar el percentatge d'encerts
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def training(dataset_dir, model_path="cnn.keras", img_size=32, epochs=20, batch_size=32, test_size=0.2):
    """
    Entrena la CNN amb matrius Canny 32x32.
    El label s'obté del primer caràcter del nom del fitxer .npy.

    Paràmetres:
        dataset_dir (str): directori on es troben les matrius .npy.
        model_path (str): ruta on es guardarà el millor model entrenat.
        img_size (int): mida de les imatges d'entrada.
        epochs (int): nombre màxim de passades completes pel dataset durant l'entrenament.
        batch_size (int): nombre d'imatges processades abans d'actualitzar els pesos del model.
        test_size (float): proporció de dades reservades per validació/test.

    Retorna:
        model: model CNN entrenat.
        history: historial de l'entrenament amb loss i accuracy per epoch.
        acc: accuracy obtinguda sobre el conjunt de test.
        class_to_id: diccionari de conversió de classe a identificador.
        id_to_class: diccionari de conversió d'identificador a classe.
    """

    X, y, class_to_id, id_to_class = load_canny_dataset(dataset_dir, img_size)

    # Dividim les dades en entrenament i test
    # test_size indica el percentatge reservat per validar
    # random_state=42 fa que la divisió sigui sempre igual
    # stratify manté la proporció de classes en train i test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=np.argmax(y, axis=1))

    model = build_cnn(img_size, num_classes=y.shape[1])

    callbacks = [
        # Atura l'entrenament si la val_loss no millora durant 5 epochs consecutives i restaura els pesos del millor model trobat
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        # Guardem el millor model basant-nos en la val_accuracy
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    # Inicia l'entrenament del model amb les dades de train i valida amb les dades de test
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)  # Avalua el model final sobre el conjunt de test

    print(f"Model guardat a: {model_path}")
    print(f"Classes detectades: {class_to_id}")
    print(f"Accuracy test: {acc:.4f}")

    return model, history, acc, class_to_id, id_to_class


def preprocess_test_image(image_path, img_size=32):
    """
    Carrega una imatge o matriu Canny i la prepara per a la CNN.
    Accepta .npy o imatges .png/.jpg.

    Paràmetres:
        image_path (str): ruta del fitxer .npy que es vol classificar.
        img_size (int): mida a la qual es redimensionarà la imatge.

    Retorna:
        img (np.ndarray): imatge preprocessada amb forma (1, img_size, img_size, 1).
    """

    if image_path.endswith(".npy"):
        img = np.load(image_path)
    else:
        return

    if img is None:
        raise ValueError(f"No s'ha pogut llegir la imatge: {image_path}")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)  # Redimensiona la matriu a la mida esperada pel model
    img = (img > 0).astype("float32") # Binaritza la imatge i la converteix a float32.

    img = img.reshape(1, img_size, img_size, 1)

    return img


def test(image_path, model_path="cnn_canny.keras", labels_path="labels.json", img_size=32):
    """
    Prediu quin caràcter hi ha en una matriu Canny.

    Paràmetres:
        image_path (str): ruta de la matriu .npy que es vol classificar.
        model_path (str): ruta del model entrenat guardat.
        labels_path (str): ruta del fitxer JSON amb el diccionari id_to_class.
        img_size (int): mida usada per preprocessar la imatge.

    Retorna:
        character (str): caràcter predit pel model.
        confidence (float): probabilitat associada a la predicció.
    """
    model = load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    id_to_class = labels_data["id_to_class"]  # Obté el diccionari que converteix IDs numèrics en caràcters.

    img = preprocess_test_image(image_path, img_size)

    prediction = model.predict(img, verbose=0)[0]  # Calcula la predicció del model, una llista de probabilitats per cada classe

    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])  # Guardem la probabilitat associada a la classe escollida.
    character = id_to_class[str(class_id)]  # Convertim l'identificador de classe al caràcter corresponent

    return character, confidence


def test(image_path, model_path="cnn_canny.keras", labels_path="labels.json", img_size=32):
    """
    Prediu quin caràcter hi ha en una matriu Canny.
    """

    model = load_model(model_path)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    id_to_class = labels_data["id_to_class"]

    img = preprocess_test_image(image_path, img_size)

    prediction = model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(prediction))
    confidence = float(prediction[class_id])

    character = id_to_class[str(class_id)]

    return character, confidence


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getcwd(), "resultats_segmentacio", "matrius")

    model, history, acc, class_to_id, id_to_class = training(dataset_dir=dataset_dir)

