import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionResNetV2
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping





class WLLMModel:
    def __init__(self, dataset_dir, output_dir):
        """
        Initializes the model with a dataset directory and sets the number of classes.
        """
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.test_dir = os.path.join(dataset_dir, 'test')

        self.class_names = sorted(os.listdir(self.train_dir))
        self.num_classes = len(self.class_names)
        base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        x = base_model.output
        #This will act as the embedding layer we will use to extract the embedding for each person
        embedding_layer_name= "embedding"
        embedding_layer = layers.Flatten(name=embedding_layer_name)(x)
        print(f"Model Shape at Additional Layer 1 {x.shape}")
        # Add a dense layer for classification (softmax output for n classes)
        class_output = Dense(self.num_classes, activation='softmax', name='classification')(embedding_layer)

        # Freeze the base model (optional, for fine-tuning later)
        for layer in base_model.layers:
            layer.trainable = False
        
        for layer in base_model.layers[-15:]:
            layer.trainable = True

        # Create the model for classification
        print(f"Base model input : {base_model.input}")
        self.model = Model(inputs=base_model.input, outputs=class_output)

        #Creat ethe model for embedding
        self.embedding_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer(embedding_layer_name).output)


        # Compile the model
        #self.model.compile(optimizer=Adam(), loss={'classification': 'categorical_crossentropy', 'embedding_layer': 'mean_squared_error'}, metrics=['accuracy'])
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        file_path = os.path.join(output_dir, 'ModelSummary.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def train_model(self, output_dir, epochs=10, batch_size=32):
        """
        Train the model using the dataset in the folder structure: 'train' and 'test'.
        Also save the best model and class names used for training as a CSV file.
        Additionally, calculate performance metrics and plot graphs.
        """
        # ImageDataGenerators for loading images
        train_datagen = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=30, 
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Load training and validation data
        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            target_size=(299, 299),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(self.test_dir,
                                                                target_size=(299, 299),
                                                                batch_size=batch_size,
                                                                class_mode='categorical')

        # Save class names used for training
        class_names = train_generator.class_indices  # This gives a dictionary of class names to indices
        class_names_list = list(class_names.keys())
        
        # Convert class names and indices to a pandas DataFrame
        class_names_df = pd.DataFrame(list(class_names.items()), columns=["Class Name", "Class Index"])
        
        model_name = os.path.basename(self.dataset_dir)
        # Save class names as a CSV file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        class_names_filepath = os.path.join(output_dir, f'{model_name}.csv')
        class_names_df.to_csv(class_names_filepath, index=False)

        # Define the callback to save the best model during training
        checkpoint = ModelCheckpoint(
            os.path.join(output_dir, f'best_model.keras'),  # Save the best model as .h5 file
            monitor='val_loss',  # Monitor validation accuracy
            verbose=1,  # Print out information when saving the model
            save_best_only=True,  # Only save the model if it's the best
            mode='min'  # Maximize validation accuracy
        )

        # Define EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',         # Monitor validation loss
            patience=15,                 # Stop after 3 epochs without improvement
            restore_best_weights=True,  # Restore the best model weights after training
            verbose=1                   # Print a message when training stops early
        )

        train_start_time = datetime.now()
        # Train the model
        history = self.model.fit(
            train_generator, 
            validation_data=validation_generator, 
            epochs=epochs, 
            callbacks=[checkpoint, early_stopping]  # Include the checkpoint callback
        )
        try:
            train_total_time = datetime.now() - train_start_time
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Save metrics to a pandas DataFrame
            train_time_data = {"Training Classes" : len(class_names), "TrainingTime":train_total_time}
            report_df = pd.DataFrame([train_time_data])
            report_df.to_csv(os.path.join(output_dir, f'TrainingTimeResults.csv'))

            # Plot training & validation accuracy and loss
            self.plot_training_history(history, output_dir=output_dir)


            val_pred = self.model.predict(validation_generator)
            val_pred_classes = np.argmax(val_pred, axis=1)


            try:
                # Calculate classification metrics (accuracy, precision, recall, f1-score)
                #true_labels = validation_generator.classes
                #report = classification_report(true_labels, val_pred_classes, target_names=class_names_list, output_dict=True)
                #print("Classification Report:\n", report)
                #Calculate classification metrics (accuracy, precision, recall, f1-score)
                true_labels = validation_generator.classes
                print(f"True Labels : {true_labels}")
                report = {}    
                report = classification_report(true_labels, val_pred_classes, target_names=class_names_list, output_dict=True)
                # Print classification reports for each head
                print(f"Classification Report for Output:\n", report)

                # Save metrics to a pandas DataFrame for each head
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(output_dir, f'classification_report.csv'))


                timestamp = datetime.now().strftime("%m-%d-%H-%M")
                # Save results to CSV
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Save metrics to a pandas DataFrame
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(output_dir, f'classification_report-{timestamp}.csv'))
            except:
                print("ERROR IN CREATING CLASSIFICATION REPORT")


            try:
                # Plot confusion matrix
                self.plot_confusion_matrix(true_labels, val_pred_classes, class_names_list, output_dir=output_dir)
            except:
                print("error in plotting confusion matrix")

        except:
            print("Error plotting graphs and saving csv")
        return history
    

    def plot_training_history(self, history, output_dir):
        """
        Plot the training and validation accuracy and loss.
        """
        # Plot training & validation accuracy
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/TrainingAndValidationPlot.png', dpi=300, bbox_inches="tight")
        plt.show()

    def plot_confusion_matrix(self, true_labels, pred_labels, class_names, output_dir):
        """
        Plot the confusion matrix using seaborn.
        """
        cm = confusion_matrix(true_labels, pred_labels)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(f'{output_dir}/ConfusionMatrix.csv', index=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{output_dir}/ConfusionMatrixPlot.png', dpi=300, bbox_inches="tight")
        plt.show()
    

    def create_embeddings_for_personality(self, data_dir, save_embeddings_dir, should_augment=False, augmentation_count=5):
        """
        Generate embeddings for all images in the provided dataset directory (train) and save the embeddings for each class (folder).
        
        Parameters:
        - data_dir: The root directory where the 'train' folder is located.
        - save_embeddings_dir: Directory where the generated embeddings will be saved.
        - embedding_dimension: Dimension of the embedding vector (default 128).
        - should_augment: Whether to apply augmentation to the images (default True).
        - augmentation_count: Number of augmented images to generate per original image (default 5).
        """
        print(f"Generating embeddings for dataset in {data_dir}")

        # Initialize the ImageDataGenerator for augmentations
        datagen = ImageDataGenerator(
            rotation_range=30,          # Random rotation between -30 and 30 degrees
            width_shift_range=0.2,      # Horizontal shift
            height_shift_range=0.2,     # Vertical shift
            shear_range=0.2,            # Shear transformation
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,       # Flip images randomly horizontally
            fill_mode='nearest'         # How to fill missing pixels after transformations
        )

        # Prepare to save embeddings
        if not os.path.exists(save_embeddings_dir):
            os.makedirs(save_embeddings_dir)

        # Iterate through each class folder (personality)
        for folder_name in os.listdir(os.path.join(data_dir, 'train')):
            folder_path = os.path.join(data_dir, 'train', folder_name)
            if os.path.isdir(folder_path):
                folder_embeddings = []

                # Process each image in the folder (class)
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    img = load_img(image_path, target_size=(299, 299))  # InceptionResNetV2 input size
                    img = img_to_array(img)  # Convert image to array
                    img = img / 255.0  # Normalize image to [0, 1]

                    if should_augment:
                        # Apply augmentation
                        augmented_images = datagen.flow(np.expand_dims(img, axis=0), batch_size=1)

                        # Generate augmented images and get their embeddings
                        augmented_embeddings = []
                        for _ in range(augmentation_count):
                            augmented_img = next(augmented_images)[0]
                            print(augmented_img.shape)
                            embedding = self.embedding_model.predict(augmented_img)  # Get embedding from the second output
                            augmented_embeddings.append(embedding)

                        # Average embeddings for this image
                        embedding = np.mean(np.array(augmented_embeddings), axis=0)
                    else:
                        embedding = self.embedding_model.predict(np.expand_dims(img, axis=0))  # Get embedding for original image

                    folder_embeddings.append(embedding)

                # Save the embeddings for the current folder (class)
                folder_embeddings = np.array(folder_embeddings)
                print(f"Folder name : {folder_name} :: Embedding Shape : {embedding.shape}")
                embedding_file = os.path.join(save_embeddings_dir, f"{folder_name}_embedding.npy")
                np.save(embedding_file, folder_embeddings)

        print(f"Embeddings saved to {save_embeddings_dir}")

    def predict_single_image(self, image_path):
        """
        Predict the class of a single image using the classification head.
        """
        img = load_img(image_path, target_size=(299, 299))  # InceptionV3 input size
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0  # Normalize image to [0, 1]

        # Expand dimensions to match the batch shape expected by the model
        img = np.expand_dims(img, axis=0)

        # Get prediction from the classification head
        predictions = self.model.predict(img)  # The first output is the classification

        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions)

        # Get the class name (folder name) from class_names
        predicted_class_name = self.class_names[predicted_class_idx]

        return predicted_class_name, predictions

    def save_model(self, model_path):
        """
        Save the model to the specified path.
        """
        self.model.save(model_path)

    def load_custom_model(self, model_path):
        """
        Load a saved model from the specified path.
        """
        self.model = load_model(model_path)

        
