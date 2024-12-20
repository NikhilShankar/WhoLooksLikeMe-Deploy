from scipy.spatial.distance import cosine
import numpy as np
import boto3
import tempfile
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

class SimilarityCalculatorAdvancedCorrected2:
    def __init__(self, embeddings_s3_path, s3_client=None):
        self.embeddings_s3_path = embeddings_s3_path
        self.s3_client = s3_client or boto3.client("s3")
        self.embeddings = self.load_embeddings()
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

    def load_embeddings(self):
        """
        Loads the saved embeddings from S3.
        """
        bucket, prefix = self._parse_s3_path(self.embeddings_s3_path)
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        embeddings = {}
        for obj in response.get("Contents", []):
            if obj["Key"].endswith("_embedding.npy"):
                folder_name = obj["Key"].split("/")[-1].replace("_embedding.npy", "")
                with tempfile.NamedTemporaryFile() as temp_file:
                    self.s3_client.download_file(bucket, obj["Key"], temp_file.name)
                    embeddings[folder_name] = np.load(temp_file.name)
        return embeddings

    def _parse_s3_path(self, s3_path):
        """
        Splits an S3 path into bucket and key.
        """
        s3_components = s3_path.replace("s3://", "").split("/", 1)
        return s3_components[0], s3_components[1]

    def preprocess_image(self, image_path):
        """
        Preprocesses the image, generates an embedding for the image.
        """
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0
        return img
