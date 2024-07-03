# simbolicke konstante
DATASET = './simpsons_dataset/'
FILTERED_DATASET = './filtered_simpsons_dataset/'
TESTSET = './simpsons_testset'
IMG_SIZE = (64, 64)
BATCH_SIZE = 100
EPOCHS = 50





# ============== Prikaz broja slika po klasama ============== #

import os
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory

dataset = image_dataset_from_directory(DATASET)

classes = dataset.class_names
print("classes:", classes)

class_image_counts = {}

for class_name in classes:
    class_dir = os.path.join(DATASET, class_name)
    num_images = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    class_image_counts[class_name] = num_images
    print(f"Klasa '{class_name}' ima {num_images} slika.")
print("\n")

# priprema podataka za vizualizaciju
class_names = list(class_image_counts.keys())
image_counts = list(class_image_counts.values())

plt.figure(figsize=(16, 16))
plt.bar(class_names, image_counts, color='skyblue')
plt.xticks(rotation='vertical')
plt.title('Broj slika po klasi u Simpsonovima datasetu')
plt.xlabel('Klase')
plt.ylabel('Broj slika')
plt.show()





# ============== Prikaz broja slika po klasama (bez klasa koje imaju manje od 5% slika nego klasa sa najvise slika) ============== #

import os
import shutil
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory

class_image_counts = {}

for class_name in classes:
    class_dir = os.path.join(DATASET, class_name)
    num_images = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    class_image_counts[class_name] = num_images

# max_images = max(class_image_counts.values())
# min_images_threshold = max_images * 0.05
#
# filtered_classes = [class_name for class_name, count in class_image_counts.items() if count >= min_images_threshold]
# print("Klase koje ostaju nakon filtriranja: ", filtered_classes)

total_images = sum(class_image_counts.values())
min_images_threshold = total_images * 0.05

filtered_classes = [class_name for class_name, count in class_image_counts.items() if count >= min_images_threshold]
print("Klase koje ostaju nakon filtriranja: ", filtered_classes)

if not os.path.exists(FILTERED_DATASET):
    os.makedirs(FILTERED_DATASET)

for class_name in filtered_classes:
    original_class_dir = os.path.join(DATASET, class_name)
    new_class_dir = os.path.join(FILTERED_DATASET, class_name)
    if not os.path.exists(new_class_dir):
        os.makedirs(new_class_dir)
    for file_name in os.listdir(original_class_dir):
        source = os.path.join(original_class_dir, file_name)
        destination = os.path.join(new_class_dir, file_name)
        if os.path.isfile(source):
            shutil.copy(source, destination)

dataset_filtered = image_dataset_from_directory(FILTERED_DATASET)

classes = dataset_filtered.class_names
class_image_counts_filtered = {}
for class_name in classes:
    class_dir = os.path.join(FILTERED_DATASET, class_name)
    num_images = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    class_image_counts_filtered[class_name] = num_images
    print(f"Klasa '{class_name}' ima {num_images} slika nakon filtriranja.")
print("\n")

class_names_filtered = list(class_image_counts_filtered.keys())
image_counts_filtered = list(class_image_counts_filtered.values())

plt.figure(figsize=(16, 16))
plt.bar(class_names_filtered, image_counts_filtered, color='skyblue')
plt.xticks(rotation='vertical')
plt.title('Broj slika po klasi u filtriranom Simpsonovima datasetu')
plt.xlabel('Klase')
plt.ylabel('Broj slika')
plt.show()





# ============== Prikaz broja slika po klasama (oversamplovane i undersamplovane klase tako da svaka klasa ima isti broj slika) ============== #

import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

target_num_images = 2000

for class_name in filtered_classes:
    class_dir = os.path.join(FILTERED_DATASET, class_name)
    image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    current_num_images = len(image_files)

    while current_num_images < target_num_images:
        random_image_file = np.random.choice(image_files)
        image = load_img(random_image_file)
        x = img_to_array(image)
        x = x.reshape((1,) + x.shape)

        for _ in datagen.flow(x, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='jpeg'):
            break

        current_num_images += 1

    if current_num_images > target_num_images:
        files_to_remove = np.random.choice(image_files, size=(current_num_images - target_num_images), replace=False)
        for file_path in files_to_remove:
            os.remove(file_path)

classes = [d for d in os.listdir(FILTERED_DATASET) if os.path.isdir(os.path.join(FILTERED_DATASET, d))]
class_image_counts = {}
for class_name in classes:
    class_dir = os.path.join(FILTERED_DATASET, class_name)
    num_images = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
    class_image_counts[class_name] = num_images
for class_name, num_images in class_image_counts.items():
    print(f"Klasa '{class_name}' ima {num_images} slika nakon augmentacije.")
print("\n")

class_names = list(class_image_counts.keys())
image_counts = list(class_image_counts.values())

plt.figure(figsize=(16, 16))
plt.bar(class_names, image_counts, color='skyblue')
plt.xticks(rotation='vertical')
plt.title('Broj slika po klasi u balansiranom Simpsonovima dataset-u')
plt.xlabel('Klase')
plt.ylabel('Broj slika')
plt.show()





# ============== Prikazati po jedan primer podatka iz svake klasu. ============== #

import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

classes = [d for d in os.listdir(FILTERED_DATASET) if os.path.isdir(os.path.join(FILTERED_DATASET, d))]

plt.figure(figsize=(10, 16))

for i, class_name in enumerate(classes):
    class_dir = os.path.join(FILTERED_DATASET, class_name)
    image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

    if image_files:
        image_file = np.random.choice(image_files)
        img_path = os.path.join(class_dir, image_file)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        plt.subplot((len(classes) + 2) // 3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        plt.xlabel(class_name)

plt.tight_layout()
plt.show()





# ============== Podeliti podatke na odgovarajuce skupove. ============== #

import os
import shutil
from keras.utils import image_dataset_from_directory

def create_and_move_to_test_set(main_path, test_set_path, num_images_per_class=400):
    if not os.path.exists(test_set_path):
        os.makedirs(test_set_path)

    classes = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]

    for class_name in classes:
        class_dir = os.path.join(main_path, class_name)
        test_class_dir = os.path.join(test_set_path, class_name)

        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        if len(image_files) < num_images_per_class:
            print(f"Nedovoljno slika u klasi '{class_name}' za premestanje. Potrebno: {num_images_per_class}, dostupno: {len(image_files)}.")
            continue

        selected_images = np.random.choice(image_files, num_images_per_class, replace=False)

        for image_file in selected_images:
            source_path = os.path.join(class_dir, image_file)
            destination_path = os.path.join(test_class_dir, image_file)
            shutil.move(source_path, destination_path)
        print(f"Premesteno {num_images_per_class} slika za klasu '{class_name}' u testni set i obrisano iz originalnog dataset-a.")
print("\n")

create_and_move_to_test_set(FILTERED_DATASET, TESTSET)

Xtrain = image_dataset_from_directory(FILTERED_DATASET,
                                      subset='training',
                                      validation_split=0.125,
                                      image_size=IMG_SIZE,
                                      batch_size=BATCH_SIZE,
                                      seed=123)

Xval = image_dataset_from_directory(FILTERED_DATASET,
                                    subset='validation',
                                    validation_split=0.125,
                                    image_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    seed=123)

Xtest = image_dataset_from_directory(TESTSET)


total_train_images = sum(1 for _ in Xtrain.unbatch())
print(f"Ukupno slika u trening skupu: {total_train_images}")

total_val_images = sum(1 for _ in Xval.unbatch())
print(f"Ukupno slika u validacionom skupu: {total_val_images}")

total_test_images = sum(1 for _ in Xtest.unbatch())
print(f"Ukupno slika u testnom skupu: {total_test_images}")


classes = Xtrain.class_names





# ============== Izvrsiti predprocesiranje podataka (skaliranje, normalizacija, ...) ============== #

from keras import Sequential
from keras import layers

data_augmentation = Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Resizing(height=64, width=64)
])





# ============== Formirati i obuciti neuralnu mrezu za resavanje datog problema. ============== #

from keras import Sequential
from keras import layers
from keras.optimizers.legacy import Adam
from keras.losses import SparseCategoricalCrossentropy

model = Sequential([
    data_augmentation,

    # layers.Conv2D(32, (3, 3), padding='same', input_shape=IMG_SIZE, activation="relu"),
    # layers.Conv2D(32, (3, 3), activation="relu"),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.2),
    #
    # layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
    # layers.Conv2D(64, (3, 3), activation="relu"),
    # layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(256, (3, 3), padding='same', activation="relu"),
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(512, (3, 3), padding='same', activation="relu"),
    layers.Conv2D(512, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

model.summary()

model.compile(Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')





# ============== primeniti neke od tehnika zastite od preobucavanja na formiran model neuralne mreze. ============== #

from keras.callbacks import EarlyStopping

callback = EarlyStopping(
    patience=5,
    monitor="val_accuracy",
    mode="max"
)





# ============== Za finalno obucen model prikazati ============== #

history = model.fit(
    Xtrain,
    epochs=EPOCHS,
    validation_data=Xval,
    callbacks=callback,
    verbose=1
)



# ============== Grafik performanse neuralne mreze kroz epohe obucavanja nad trening i validacionom skupu, ============== #

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()

plt.subplot(121)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.show()



# ============== Matricu konfucije na trening i test skupu, ============== #

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(dataset, dataset_name):
    labels = np.array([])
    pred = np.array([])

    for img, lab in dataset:
        labels = np.append(labels, lab.numpy())
        pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

    accuracy = 100 * accuracy_score(labels, pred)
    print(f'Tačnost modela na {dataset_name} skupu je: {accuracy:.2f}%')

    cm = confusion_matrix(labels, pred, normalize='true')
    fig, ax = plt.subplots(figsize=(16, 16))
    cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    cmDisplay.plot(ax=ax)
    plt.title(f'Konfuziona matrica za {dataset_name} skup')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

evaluate_model(Xtrain, 'trening')
evaluate_model(Xtest, 'test')



# ============== Primere dobro i lose klasifikovanih primera dataset-a. ============== #

def evaluate_model_and_show_examples(dataset, dataset_name):
    correct_imgs = []
    incorrect_imgs = []
    correct_labels = []
    incorrect_labels = []
    predicted_labels_correct = []
    predicted_labels_incorrect = []

    for img_batch, label_batch in dataset:
        preds = model.predict(img_batch, verbose=0)
        pred_labels = np.argmax(preds, axis=1)

        for i in range(len(pred_labels)):
            if pred_labels[i] == label_batch[i]:
                if len(correct_imgs) < 4:
                    correct_imgs.append(img_batch[i].numpy())
                    correct_labels.append(label_batch[i].numpy())
                    predicted_labels_correct.append(pred_labels[i])
            else:
                if len(incorrect_imgs) < 4:
                    incorrect_imgs.append(img_batch[i].numpy())
                    incorrect_labels.append(label_batch[i].numpy())
                    predicted_labels_incorrect.append(pred_labels[i])

    plt.figure(figsize=(14, 8))
    for i in range(4):
        if i < len(correct_imgs):
            plt.subplot(2, 4, i+1)
            plt.imshow(correct_imgs[i].astype("uint8"))
            plt.title(f"True: {classes[correct_labels[i]]}\nPred: {classes[predicted_labels_correct[i]]}")
            plt.axis('off')

        if i < len(incorrect_imgs):
            plt.subplot(2, 4, i+5)
            plt.imshow(incorrect_imgs[i].astype("uint8"))
            plt.title(f"True: {classes[incorrect_labels[i]]}\nPred: {classes[predicted_labels_incorrect[i]]}")
            plt.axis('off')
    plt.suptitle(f"Correct and Incorrect Predictions on {dataset_name} Set")
    plt.show()

evaluate_model_and_show_examples(Xtest, 'test')
