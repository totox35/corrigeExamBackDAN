from transformers import AutoModel

import onnxruntime as ort
import numpy as np
import json
import threading
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class EmbeddingController:
    def __init__(self, model_path: str):
        # Initialize the ONNX runtime session
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
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
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #model_path = "src/main/embedding/model.onnx"
    #controller = EmbeddingController(model_path)

    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True, device_map="cpu")

    texts = input()
    texts = json.loads(texts)

    embeddings = []
    for text in texts:
        embedding = model.encode([text], task="text-matching")
        embedding = embedding.flatten()
        embedding = embedding.tolist()
        embeddings.append(embedding)
    print(json.dumps(embeddings))

if __name__ == "__main__":
    main()
