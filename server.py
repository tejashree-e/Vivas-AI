import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir = r"C:\Desktop\proj\Final_dataset\train"
valid_dir = r"C:\Desktop\proj\Final_dataset\valid"
test_dir = r"C:\Desktop\proj\Final_dataset\test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),  # Explicit Input layer
    
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20
)

model.save("crop_disease_model.h5")

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

def predict_disease(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_name = class_labels[predicted_class]
    
    disease_info = {
        "Pepper__bell___Bacterial_spot": ("Bacterial Spot", "A bacterial infection causing dark, water-soaked spots on leaves and fruits, leading to reduced yield."),
        "Pepper__bell___healthy": ("Healthy Pepper Plant", "No visible disease symptoms, vibrant green leaves, and normal fruit development."),
        "Potato___Early_blight": ("Early Blight", "Dark brown spots with concentric rings on leaves caused by Alternaria solani. Weakens the plant and reduces tuber size."),
        "Potato___healthy": ("Healthy Potato Plant", "No signs of disease, with strong foliage and uniform tuber growth."),
        "Potato___Late_blight": ("Late Blight", "Deadly fungal disease (Phytophthora infestans) causing dark, water-soaked lesions on leaves and tubers."),
        "Rice_BACTERIAL LEAF BLIGHT": ("Bacterial Leaf Blight", "Water-soaked streaks on leaves turning yellow and brown, caused by Xanthomonas oryzae."),
        "Rice_BROWN SPOT": ("Brown Spot", "Brown lesions with yellow halos on leaves and grains caused by Cochliobolus miyabeanus."),
        "Rice_DEFICIENCY- MAGNESIUM": ("Magnesium Deficiency", "Yellowing between leaf veins, stunted growth, and poor grain filling."),
        "Rice_DEFICIENCY- NITROGEN": ("Nitrogen Deficiency", "Pale yellowing leaves and reduced tillering due to lack of nitrogen."),
        "Rice_DEFICIENCY- NITROGEN MANGANESE POTASSIUM MAGNESIUM and ZINC": ("Multiple Nutrient Deficiency", "Stunted growth, discoloration, weak stems, and poor grain formation."),
        "Rice_DISEASE- Narrow Brown Spot NUTRIENT DEFFICIENT- Nitrogen -N- Potassium -K- Calcium -Ca-": ("Narrow Brown Spot", "Small, dark brown spots with yellow halos affecting leaves, often linked to nitrogen and potassium deficiencies."),
        "Rice_DISEASE- Bacterial Leaf Blight NUTRIENT DEFFICIENT- Silicon": ("Bacterial Leaf Blight & Silicon Deficiency", "Combination of bacterial streaks and poor resistance due to silicon deficiency."),
        "Rice_DISEASE- Hispa NUTRIENT DEFFICENCY- N-A - Integrated pest management practices-": ("Hispa", "Leaf scraping damage by Hispa beetles, causing parallel white streaks, worsened by nutrient deficiencies."),
        "Rice_DISEASE- Lead Scald NUTRIENT DEFFICIENT- Nitrogen -N- Potassium -K- Calcium -Ca- Sulfur -S-": ("Leaf Scald", "Long reddish-brown lesions, exacerbated by nitrogen, potassium, calcium, and sulfur deficiencies."),
        "Rice_DISEASE- Leaf Blast NUTRIENT DEFFICIENT- Silicon- Nitrogen -N- Potassium -K- Potassium -K- Calcium -Ca-": ("Leaf Blast", "White to gray spots on leaves that expand into lesions, worsened by multiple nutrient deficiencies."),
        "Rice_HEALTHY": ("Healthy Rice Plant", "No signs of disease, strong green leaves, and normal growth."),
        "Rice_HISPA": ("Rice Hispa", "Damage by Hispa beetles, causing white parallel lines and skeletonized leaves."),
        "Rice_LEAFBLAST": ("Leaf Blast", "Gray-green lesions on leaves that enlarge into spindle-shaped spots, leading to severe yield loss."),
        "SC_Bacterial Blight": ("Sugarcane Bacterial Blight", "Leaf scald, wilting, and white streaks on leaves caused by Xanthomonas albilineans."),
        "SC_BrownRust": ("Brown Rust", "Fungal disease causing reddish-brown pustules on leaves, reducing photosynthesis and yield."),
        "SC_Dried Leaves": ("Dried Leaves", "Physiological drying due to aging, water stress, or disease."),
        "SC_Healthy": ("Healthy Sugarcane Plant", "Lush green leaves, no visible disease symptoms, and strong stalks."),
        "SC_Mawa": ("Mawa Disease", "A viral disease causing severe stunting and yellowing of sugarcane plants."),
        "SC_Mites": ("Mite Infestation", "Tiny mites sucking plant sap, leading to yellowing and poor growth."),
        "SC_Red Rot": ("Red Rot", "Severe fungal disease (Colletotrichum falcatum) causing internal red discoloration of stems, leading to rotting."),
        "SC_RedSpot": ("Red Spot", "Small reddish lesions appearing on leaves, affecting plant health."),
        "SC_YellowLeaf": ("Yellow Leaf Disease", "Yellowing of leaves due to a viral infection, reducing sugar content in canes."),
        "Tomato__Target_Spot": ("Target Spot", "Circular spots with grayish centers on leaves caused by Corynespora cassiicola."),
        "Tomato__Tomato_mosaic_virus": ("Tomato Mosaic Virus", "Mosaic-like yellow and green mottling on leaves, affecting fruit quality."),
        "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Yellow Leaf Curl Virus", "Yellow, curled leaves and stunted growth caused by a viral infection spread by whiteflies."),
        "Tomato_Bacterial_spot": ("Bacterial Spot", "Small, dark spots on leaves and fruits, caused by Xanthomonas spp."),
        "Tomato_Early_blight": ("Early Blight", "Dark spots with concentric rings caused by Alternaria solani, leading to premature leaf drop."),
        "Tomato_healthy": ("Healthy Tomato Plant", "No disease symptoms, vibrant green leaves, and uniform fruit development."),
        "Tomato_Late_blight": ("Late Blight", "A devastating fungal disease (Phytophthora infestans) causing dark, wet lesions and rapid plant death."),
        "Tomato_Leaf_Mold": ("Leaf Mold", "Yellow spots on upper leaves with mold growth underneath, caused by Passalora fulva."),
        "Tomato_Septoria_leaf_spot": ("Septoria Leaf Spot", "Numerous small, dark spots with light centers, leading to leaf drop."),
        "Tomato_Spider_mites_Two_spotted_spider_mite": ("Spider Mite Infestation", "Tiny arachnids sucking plant sap, causing yellow speckling and webbing."),
    }
    
    if class_name in disease_info:
        disease_name, description = disease_info[class_name]
    else:
        disease_name = "No Matching Disease Found"
        description = "The uploaded image does not match any known disease in the dataset."
    
    return {"Disease Name": disease_name, "Description": description}