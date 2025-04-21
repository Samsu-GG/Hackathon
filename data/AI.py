import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths and parameters
data_dir = 'waste_images/'  # Update with your image directory
img_height, img_width = 224, 224  # VGG models expect 224x224 images
batch_size = 32
epochs = 20
num_classes = 3  # compost, recycle, trash

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# Function to create a model with transfer learning
def create_model(base_model_type='vgg16'):
    if base_model_type.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    elif base_model_type.lower() == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    else:
        raise ValueError("base_model_type must be 'vgg16' or 'vgg19'")

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification head
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create and train VGG16 model
model_vgg16 = create_model('vgg16')
print("VGG16 Model Summary:")
model_vgg16.summary()

# Define callbacks
checkpoint_vgg16 = ModelCheckpoint(
    'waste_classifier_vgg16.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the VGG16 model
history_vgg16 = model_vgg16.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_vgg16, early_stopping]
)

# Create and train VGG19 model
model_vgg19 = create_model('vgg19')
print("VGG19 Model Summary:")
model_vgg19.summary()

# Define checkpoint for VGG19
checkpoint_vgg19 = ModelCheckpoint(
    'waste_classifier_vgg19.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the VGG19 model
history_vgg19 = model_vgg19.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_vgg19, early_stopping]
)


# Function to plot training history
def plot_history(history_vgg16, history_vgg19):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history_vgg16.history['accuracy'], label='VGG16 Train')
    ax1.plot(history_vgg16.history['val_accuracy'], label='VGG16 Val')
    ax1.plot(history_vgg19.history['accuracy'], label='VGG19 Train')
    ax1.plot(history_vgg19.history['val_accuracy'], label='VGG19 Val')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # Plot loss
    ax2.plot(history_vgg16.history['loss'], label='VGG16 Train')
    ax2.plot(history_vgg16.history['val_loss'], label='VGG16 Val')
    ax2.plot(history_vgg19.history['loss'], label='VGG19 Train')
    ax2.plot(history_vgg19.history['val_loss'], label='VGG19 Val')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


# Plot training history
plot_history(history_vgg16, history_vgg19)

# Evaluate both models
print("Evaluating VGG16 model:")
vgg16_scores = model_vgg16.evaluate(validation_generator)
print(f"VGG16 - Loss: {vgg16_scores[0]}, Accuracy: {vgg16_scores[1]}")

print("Evaluating VGG19 model:")
vgg19_scores = model_vgg19.evaluate(validation_generator)
print(f"VGG19 - Loss: {vgg19_scores[0]}, Accuracy: {vgg19_scores[1]}")


# Function to generate confusion matrix
def evaluate_model(model, generator, model_name):
    # Get true labels
    true_labels = generator.classes

    # Get predictions
    generator.reset()
    predictions = model.predict(generator)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Print classification report
    class_names = list(generator.class_indices.keys())
    report = classification_report(true_labels, predicted_labels, target_names=class_names)

    print(f"\n{model_name} Classification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.show()


# Evaluate models with confusion matrix
evaluate_model(model_vgg16, validation_generator, "VGG16")
evaluate_model(model_vgg19, validation_generator, "VGG19")

# Fine-tuning (optional)
# Uncomment this section if you want to fine-tune after initial training
'''
def fine_tune_model(model, base_model_type):
    # Unfreeze some layers
    if base_model_type.lower() == 'vgg16':
        for layer in model.layers[0].layers[-4:]:
            layer.trainable = True
    elif base_model_type.lower() == 'vgg19':
        for layer in model.layers[0].layers[-5:]:
            layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Fine-tune VGG16
fine_tuned_vgg16 = fine_tune_model(model_vgg16, 'vgg16')
history_ft_vgg16 = fine_tuned_vgg16.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_vgg16, early_stopping]
)

# Fine-tune VGG19
fine_tuned_vgg19 = fine_tune_model(model_vgg19, 'vgg19')
history_ft_vgg19 = fine_tuned_vgg19.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint_vgg19, early_stopping]
)
'''


# Make predictions on new images
def predict_waste_type(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])

    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[class_idx]
    confidence = prediction[0][class_idx] * 100

    return predicted_class, confidence

# Example usage:
# predicted_class, confidence = predict_waste_type(model_vgg16, 'new_image.jpg')
# print(f"Predicted waste type: {predicted_class} with {confidence:.2f}% confidence")