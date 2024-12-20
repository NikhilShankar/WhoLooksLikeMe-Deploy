from tensorflow.keras.models import load_model, Model
import boto3
import tempfile

class WLLMModelLoader:
    def __init__(self, s3_model_path, s3_client=None):
        """
        Initializes the ModelLoader class and loads the model directly from S3.

        :param s3_model_path: S3 path to the model file.
        :param s3_client: Boto3 S3 client.
        """
        self.s3_model_path = s3_model_path
        self.s3_client = s3_client or boto3.client("s3")
        self.model = self._load_model_from_s3()
        self.embedding_model = None
        self.create_embedding_model()

    def _load_model_from_s3(self):
        """
        Downloads the model from S3 and loads it.
        """
        bucket, key = self._parse_s3_path(self.s3_model_path)
        with tempfile.NamedTemporaryFile() as temp_file:
            self.s3_client.download_file(bucket, key, temp_file.name)
            return load_model(temp_file.name)

    def _parse_s3_path(self, s3_path):
        """
        Splits an S3 path into bucket and key.
        """
        s3_components = s3_path.replace("s3://", "").split("/", 1)
        return s3_components[0], s3_components[1]

    def create_embedding_model(self, layer_name='embedding'):
        """
        Creates an embedding model from the specified layer of the loaded model.
        """
        try:
            embedding_layer = self.model.get_layer(name=layer_name)
            self.embedding_model = Model(inputs=self.model.input, outputs=embedding_layer.output)
        except ValueError:
            print(f"Error: Layer with name '{layer_name}' not found in the model.")
