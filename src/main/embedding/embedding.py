from transformers import AutoModel
import json
import threading
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)


class EmbeddingController:
    def __init__(self, model_path: str):
        # Initialize the ONNX runtime session
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, device_map='cpu')
        self.ort_session = ort.InferenceSession(model_path)
        # Cache to store embeddings
        self.cache = {}
        # Condition variable to wait for data readiness
        self.data_ready = threading.Condition()
        self.all_data_collected = False
        self.texts = []

    def signal_data_ready(self, texts):
        """
        Signal that all data is ready for processing and provide the texts.
        """
        with self.data_ready:
            self.texts = texts
            self.all_data_collected = True
            self.data_ready.notify_all()

    def wait_for_data(self):
        """
        Wait until all data is ready for processing.
        """
        with self.data_ready:
            while not self.all_data_collected:
                self.data_ready.wait()

    def get_embedding_from_txt(self, txt: str) -> list:
        """
        Converts a text to its embedding.

        Parameters:
        txt (str): A string to convert to embeddings.

        Returns:
        list: The embedding of the text.
        """
        # Check if the embedding is already in the cache
        if txt in self.cache:
            return self.cache[txt]

        inputs = {self.ort_session.get_inputs()[0].name: np.array([txt], dtype=np.float32)}
        outputs = self.ort_session.run(None, inputs)
        embedding = outputs[0].tolist()
        self.cache[txt] = embedding

        return embedding

    def get_embedding_from_txt_list(self, txt_l: list) -> list:
        """
        Converts a list of texts to their embeddings.

        Parameters:
        txt_l (list): A list of strings to convert to embeddings.

        Returns:
        list: A list of embeddings.
        """
        embeddings = []
        for text in txt_l:
            embedding = self.get_embedding_from_txt(text)
            embeddings.append(embedding)

        return embeddings

def main():
    try:
        # Load the model
        model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, device_map="cpu", truncate_dim = 1024)

        # Read input from stdin
        texts_input = sys.stdin.readline().strip()
        
        # Log received input for debugging (will be captured by Java process)
        print(f"DEBUG: Received input: {texts_input}", file=sys.stderr)
        
        # Parse the JSON input
        texts = json.loads(texts_input)

        # Process embeddings
        embeddings = []
        for text in texts:
            embedding = model.encode([text], task="text-matching")
            embedding = embedding.flatten()
            embedding = embedding.tolist()
            embeddings.append(embedding)
        
        # Format and print the result as valid JSON
        result_json = json.dumps(embeddings)
        print(result_json, flush=True)
        
    except Exception as e:
        # Log any errors for debugging
        print(f"ERROR in embedding.py: {str(e)}", file=sys.stderr)
        # Still try to return a valid JSON format with error info
        error_response = json.dumps({"error": str(e)})
        print(error_response, flush=True)


if __name__ == "__main__":
    main()