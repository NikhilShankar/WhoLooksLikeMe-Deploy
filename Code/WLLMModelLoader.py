from tensorflow.keras.models import load_model, Model

class WLLMModelLoader:
    def __init__(self, model_path):
        """
        Initializes the ModelLoader class and loads the model.
        
        :param model_path: Path to the .h5 model file.
        """
        self.model = load_model(model_path)
        self.embedding_model = None
        self.create_embedding_model()

    def create_embedding_model(self, layer_name='embedding'):
        """
        Creates an embedding model from the specified layer of the loaded model.
        
        :param layer_name: Name of the layer to extract for the embedding model.
        """
        try:
            # Get the specified layer by name
            embedding_layer = self.model.get_layer(name=layer_name)
            
            # Create a new model that outputs from the specified layer
            self.embedding_model = Model(
                inputs=self.model.input,
                outputs=embedding_layer.output
            )
            print(f"Embedding model created using layer: '{layer_name}'.")
        except ValueError:
            print(f"Error: Layer with name '{layer_name}' not found in the model.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
