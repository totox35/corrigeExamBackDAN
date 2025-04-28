#!/usr/bin/env python3
# get_related_chunks.py
import sys
import json
from elasticsearch import Elasticsearch
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize Elasticsearch client with scheme parameter
es = Elasticsearch(['http://localhost:9200'])

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    """
    Get embeddings for a list of texts using sentence-transformers.
    
    Parameters:
    texts (list): List of text strings to get embeddings for.
    
    Returns:
    list: List of embeddings (numpy arrays).
    """
    embeddings = model.encode(texts)
    return embeddings

def get_relevant_chunks(text, index_name, top_n=5):
    """
    Get the most relevant chunks from Elasticsearch based on the similarity of their embeddings.

    Parameters:
    text (str): The input text to search for relevant chunks.
    index_name (str): The name of the Elasticsearch index to query.
    top_n (int): The number of top relevant chunks to retrieve.

    Returns:
    list: The most relevant chunks from Elasticsearch.
    """
    # Get the embedding of the input text
    input_embedding = get_embeddings([text])[0]

    if len(input_embedding) != 1024:
        raise ValueError(f"Embedding dimension mismatch: got {len(input_embedding)} but expected 1024")

    
    # Convert embedding to a list and format for ES script
    query_vector = input_embedding.tolist()
    
    # Use a script score query which calculates cosine similarity
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    # Execute search
    response = es.search(
        index=index_name,
        query=script_query,
        size=top_n
    )

    # Extract relevant chunks based on cosine similarity score
    relevant_chunks = []
    for doc in response['hits']['hits']:
        chunk = doc["_source"]
        # Add the score for reference
        chunk['score'] = doc['_score']
        relevant_chunks.append(chunk)

    return relevant_chunks

def main():
    """
    Main function to handle command-line arguments and execute the search.
    """
    # Check if we have the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python get_related_chunks.py <query_text> <course_name> [top_n]")
        sys.exit(1)
    
    # Get arguments
    query_text = sys.argv[1]
    course_name = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Construct index name (assuming same format as in your other scripts)
    index_name = f"course_{course_name.lower().replace(' ', '_')}"
    
    try:
        # Get relevant chunks
        chunks = get_relevant_chunks(query_text, index_name, top_n)
        
        # Output results as JSON
        print(json.dumps(chunks, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()