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

        img = cv2.resize(
            img,
            (img_size, img_size),
            interpolation=cv2.INTER_NEAREST
        )

        img = (img > 0).astype("float32")

        label = filename[0]

        images.append(img)
        labels.append(label)

    classes = sorted(list(set(labels)))

    class_to_id = {c: i for i, c in enumerate(classes)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    y_ids = np.array([class_to_id[label] for label in labels], dtype="int32")

    X = np.array(images, dtype="float32")
    X = X.reshape(-1, img_size, img_size, 1)

    y = to_categorical(y_ids, num_classes=len(classes))

    return X, y, class_to_id, id_to_class


def build_cnn(img_size, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def training(dataset_dir, model_path="cnn.keras",
             img_size=32, epochs=20, batch_size=32, test_size=0.2):
    """
    Entrena la CNN amb matrius Canny 32x32.
    El label s'obté del primer caràcter del nom del fitxer .npy.
    """

    X, y, class_to_id, id_to_class = load_canny_dataset(dataset_dir, img_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=np.argmax(y, axis=1)
    )

    model = build_cnn(img_size, num_classes=y.shape[1])

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Model guardat a: {model_path}")
    print(f"Classes detectades: {class_to_id}")
    print(f"Accuracy test: {acc:.4f}")

    return model, history, acc, class_to_id, id_to_class


def preprocess_test_image(image_path, img_size=32):
    """
    Carrega una imatge o matriu Canny i la prepara per a la CNN.
    Accepta .npy o imatges .png/.jpg.
    """

    if image_path.endswith(".npy"):
        img = np.load(image_path)
    else:
        return

    if img is None:
        raise ValueError(f"No s'ha pogut llegir la imatge: {image_path}")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    img = (img > 0).astype("float32")
    img = img.reshape(1, img_size, img_size, 1)

    return img


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

    model, history, acc, class_to_id, id_to_class = training(
        dataset_dir=dataset_dir,
        model_path="cnn.keras",
        img_size=32,
        epochs=20,
        batch_size=32,
        test_size=0.2
    )
