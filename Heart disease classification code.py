# ================================
#  MOUNT DRIVE & EXTRACT DATA
# ================================
from google.colab import drive
drive.mount('/content/drive')

import os

ZIP_PATH = "/content/drive/MyDrive/project/Heart.zip"

print("ZIP Exists:", os.path.exists(ZIP_PATH))


import zipfile

EXTRACT_PATH = "/content/heart_extracted"

os.makedirs(EXTRACT_PATH, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)

print("Dataset Extracted Successfully")
print(os.listdir(EXTRACT_PATH))

# ================================
#  FIX NESTED FOLDERS
# ================================
import shutil

base = "/content/heart_extracted"

for cls in os.listdir(base):

    outer = os.path.join(base, cls)

    if os.path.isdir(outer):
        items = os.listdir(outer)

        if len(items) == 1 and os.path.isdir(os.path.join(outer, items[0])):

            inner = os.path.join(outer, items[0])

            for file in os.listdir(inner):
                shutil.move(os.path.join(inner, file), outer)

            os.rmdir(inner)

print("Nested Folder Fix Completed")



# ================================
#  CONVERT IMAGES TO JPG
# ================================
from PIL import Image

converted = 0

for root, dirs, files in os.walk(base):
    for file in files:

        path = os.path.join(root, file)

        if not file.lower().endswith(".jpg"):

            try:
                img = Image.open(path).convert("RGB")
                new_path = os.path.splitext(path)[0] + ".jpg"

                img.save(new_path, "JPEG")
                os.remove(path)

                converted += 1

            except:
                pass

print("Converted Images:", converted)


# ================================
#  COUNT IMAGES
# ================================
count = 0

for root, dirs, files in os.walk(base):
    for file in files:
        if file.lower().endswith(".jpg"):
            count += 1

print("TOTAL IMAGES:", count)

for cls in os.listdir(base):
    print(cls, "->", len(os.listdir(os.path.join(base, cls))))

# ============================================
#  IMPORT LIBRARIES
# ============================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# ============================================
#  PARAMETERS
# ============================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================
#  DATA SPLIT (70-12-18)
# ============================================
import random

split_base = "/content/heart_split"

train_path = os.path.join(split_base, "train")
val_path   = os.path.join(split_base, "val")
test_path  = os.path.join(split_base, "test")

for path in [train_path, val_path, test_path]:
    os.makedirs(path, exist_ok=True)

for cls in os.listdir(base):
    cls_path = os.path.join(base, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [img for img in os.listdir(cls_path) if img.lower().endswith(".jpg")]

    random.shuffle(images)

    total = len(images)

    train_end = int(0.7 * total)
    val_end   = int(0.82 * total)

    train_imgs = images[:train_end]
    val_imgs   = images[train_end:val_end]
    test_imgs  = images[val_end:]

    for split, imgs in zip(
        [train_path, val_path, test_path],
        [train_imgs, val_imgs, test_imgs]
    ):
        os.makedirs(os.path.join(split, cls), exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(split, cls, img)
            )

print("Dataset Split Completed (70-12-18)")

# ============================================
#  DATA GENERATORS
# ============================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_gen = val_test_datagen.flow_from_directory(
    val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
#  HYBRID MODEL (VGG16 + InceptionV3)
# ============================================
input_layer = Input(shape=(224, 224, 3))

vgg = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
inc = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer)

for layer in vgg.layers[:-8]:
    layer.trainable = False

for layer in inc.layers[:-20]:
    layer.trainable = False

vgg_feat = GlobalAveragePooling2D()(vgg.output)
inc_feat = GlobalAveragePooling2D()(inc.output)

combined = Concatenate()([vgg_feat, inc_feat])

x = BatchNormalization()(combined)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.6)(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)

output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(input_layer, output)


# ============================================
# COMPILE MODEL
# ============================================
model.compile(
    optimizer=Adam(5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# ============================================
#  TRAIN MODEL
# ============================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100
)

# ============================================
#  PLOT ACCURACY
# ============================================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.title("Training vs Validation Accuracy")
plt.show()


# ============================================
#  VALIDATION EVALUATION (FIXED)
# ============================================
val_gen.reset()

pred = model.predict(val_gen, steps=len(val_gen))

y_pred = np.argmax(pred, axis=1)
y_true = val_gen.classes

print("Validation Samples:", len(y_true))
print("Predicted Samples:", len(y_pred))

class_names = list(val_gen.class_indices.keys())

print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
print("Precision:", precision_score(y_true, y_pred, average='weighted'))
print("Recall:", recall_score(y_true, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_true, y_pred, average='weighted'))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))



# ============================================
#  TEST EVALUATION (FIXED)
# ============================================
test_gen.reset()

pred = model.predict(test_gen, steps=len(test_gen))

y_pred = np.argmax(pred, axis=1)
y_true = test_gen.classes

print("Test Samples:", len(y_true))
print("Predicted Samples:", len(y_pred))

loss, acc = model.evaluate(test_gen)

print("Test Accuracy:", round(acc * 100, 2), "%")

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

# ============================================
#  SINGLE IMAGE vs MODEL ACCURACY
# ============================================
import random
from tensorflow.keras.preprocessing import image

# Get class names
class_names = list(test_gen.class_indices.keys())

# Pick a random image from test set
sample_class = random.choice(class_names)
sample_folder = os.path.join(test_path, sample_class)
sample_image_name = random.choice(os.listdir(sample_folder))
sample_image_path = os.path.join(sample_folder, sample_image_name)

print("Selected Image:", sample_image_path)
print("Actual Class:", sample_class)

# Load and preprocess image
img = image.load_img(sample_image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Predicted Class:", predicted_class)
print("Confidence:", round(confidence * 100, 2), "%")

# ============================================
# PLOT COMPARISON
# ============================================
plt.figure()

# DO NOT set colors manually (per rules)
values = [acc * 100, confidence * 100]
labels = ["Test Accuracy", "Image Confidence"]

plt.bar(labels, values)

plt.title("Model Accuracy vs Single Image Confidence")
plt.ylabel("Percentage (%)")

plt.show()

# ============================================
# SHOW IMAGE
# ============================================
plt.imshow(img)
plt.title(f"Actual: {sample_class} | Predicted: {predicted_class}")
plt.axis('off')
plt.show()
