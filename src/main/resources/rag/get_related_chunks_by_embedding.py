#!/usr/bin/env python3
# get_related_chunks_by_embedding.py
import sys
import json
from elasticsearch import Elasticsearch
import numpy as np

# Initialize Elasticsearch client with scheme parameter
es = Elasticsearch(['http://localhost:9200'])

def get_relevant_chunks_by_embedding(embedding, index_name, top_n=5):
    """
    Get the most relevant chunks from Elasticsearch based on a pre-calculated embedding.

    Parameters:
    embedding (list): The pre-calculated embedding vector
    index_name (str): The name of the Elasticsearch index to query
    top_n (int): The number of top relevant chunks to retrieve

    Returns:
    list: The most relevant chunks from Elasticsearch, only including 'text' field
    """
    # Use a script score query which calculates cosine similarity
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": embedding}
            }
        }
    }
    
    # Execute search
    response = es.search(
        index=index_name,
        query=script_query,
        size=top_n
    )

    relevant_chunks = []
    for doc in response['hits']['hits']:
        chunk = doc["_source"]
        relevant_chunks.append({
            "text": chunk['text'] 
        })

    return relevant_chunks

def main():
    """
    Main function to handle command-line arguments and execute the search.
    """
    # Check if we have the correct number of arguments
    if len(sys.argv) < 3:
        print("Usage: python get_related_chunks_by_embedding.py <embedding_file_path> <course_name> [top_n]")
        sys.exit(1)
    
    # Get arguments
    embedding_file_path = sys.argv[1]
    course_name = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    try:
        # Read the embedding from the temporary file
        with open(embedding_file_path, 'r') as f:
            embedding = json.load(f)
        
        #print(embedding)
        if len(embedding) != 1024:
            raise ValueError(f"Embedding dimension mismatch: got {len(embedding)} but expected 1024")

        # Construct index name (assuming same format as in your other scripts)
        index_name = f"course_{course_name.lower().replace(' ', '_')}"
        
        # Get relevant chunks using the pre-calculated embedding
        chunks = get_relevant_chunks_by_embedding(embedding, index_name, top_n)
        
        # Output results as JSON
        print(json.dumps(chunks, indent=2))
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()