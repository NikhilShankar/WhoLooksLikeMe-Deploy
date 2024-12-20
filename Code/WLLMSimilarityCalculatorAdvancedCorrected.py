from scipy.spatial.distance import cosine
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator



class SimilarityCalculatorAdvancedCorrected2:
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.embeddings = self.load_embeddings()
        self.datagen = ImageDataGenerator(
            rotation_range=15,          # Random rotation between -30 and 30 degrees
            width_shift_range=0.2,      # Horizontal shift
            height_shift_range=0.2,     # Vertical shift
            shear_range=0.2,            # Shear transformation
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,       # Flip images randomly horizontally
            fill_mode='nearest'         # How to fill missing pixels after transformations
        )

    def load_embeddings(self):
        """
        Loads the saved embeddings from files.
        """
        embeddings = {}
        for embedding_file in os.listdir(self.embeddings_dir):
            if embedding_file.endswith("_embedding.npy"):
                folder_name = embedding_file.replace("_embedding.npy", "")
                embedding_path = os.path.join(self.embeddings_dir, embedding_file)
                embedding = np.load(embedding_path)
                embeddings[folder_name] = embedding
        return embeddings

    def preprocess_image(self, image_path):
        """
        Preprocesses the image, generates an embedding for the image.
        """
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0 
        return img

    def calculate_similarity(self, test_image_path, model, topN=5, num_iterations=10):

        # Load embeddings
        all_embeddings = {}
        for file_name in os.listdir(self.embeddings_dir):
            if file_name.endswith("_embedding.npy"):
                person = file_name.split("_embedding")[0]
                all_embeddings[person] = np.load(os.path.join(self.embeddings_dir, file_name))

        """
        Compares the test image to the saved embeddings and returns the similarity score.
        """
        print(f"{test_image_path}")
        img = load_img(test_image_path, target_size=(299, 299))
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0 
        augmented_test_images = self.datagen.flow(np.expand_dims(img, axis=0), batch_size=1)

        # Progressive narrowing using binary search style
        remaining_persons = list(all_embeddings.keys())
        print(f'Total Remaining : {len(remaining_persons)}')
        cumulative_scores = {person: 0 for person in remaining_persons}
        frequency_scores = {person: 0 for person in remaining_persons}

        currentIter = 0
        while currentIter < num_iterations:
            augmented_test_image = next(augmented_test_images)[0]
            test_embedding = model.predict(np.expand_dims(augmented_test_image, axis=0)) 
            round_scores = []
            for person in remaining_persons:
                similarities = self.calculate_cosine_similarity(test_embedding, all_embeddings[person])
                round_scores.append((person, max(similarities)))  # Choose the max similarity in this round
            currentIter+=1
            round_scores.sort(key=lambda x: x[1], reverse=True)
            for index, (person, similarity) in enumerate(round_scores):
                cumulative_scores[person] += similarity
                frequency_scores[person] += (len(round_scores) - index)

        average_scores = {person: cumulative_scores[person] / num_iterations for person in remaining_persons}
        # Debugging: Print final scores
        
        # Rank by both average score and frequency count
        prediction_by_average = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
        prediction_by_total = sorted(frequency_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("*****************************************************")
        print("*****************************************************")
        print(f"Average scores: \n{average_scores}")
        print(f"Total scores: \n{frequency_scores}")
        print("*"*50)
        print(f"Average Scores Sorted \n{prediction_by_average}")
        print(f"Frequency Scores Sorted \n{prediction_by_total}")
        print("*****************************************************\n\n")
        return prediction_by_average, prediction_by_total


    def calculate_cosine_similarity(self, test_embedding, candidate_embeddings):
        """
        Calculate cosine similarity between a test embedding and candidate embeddings.
        """
        return [1 - cosine(test_embedding.flatten(), candidate.flatten()) for candidate in candidate_embeddings]