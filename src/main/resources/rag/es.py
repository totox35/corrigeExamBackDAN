from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
import uuid
from embedding import get_embeddings
from chunking import chunk_text_from_pdf, read_pdf_file_as_binary

# Connect to local Elasticsearch
es = Elasticsearch("http://localhost:9200")

def create_index(index_name: str, dims: int):
    """
    Creates an index in the Elasticsearch client if it does not exist.

    Parameters:
    index_name (str): The index to be created.
    dims (int): The number of dimensions for the dense_vector field.
    """
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Deleted existing index {index_name}.")

    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims  # Use the provided number of dimensions
                    },
                    "pdf_name": {"type": "keyword"}  # PDF identification
                }
            }
        }
    )
    print(f"Index {index_name} created successfully with {dims} dimensions.")

def add_data(texts: list, course_name: str, pdf_name: str):
    """
    Add texts with the corresponding embedding and metadata to Elasticsearch.

    Parameters:
    texts (list): The list of texts to add.
    course_name (str): The name of the course.
    pdf_name (str): The name of the PDF from which the chunks originate.
    """
    index_name = "course_" + course_name
    embeddings = get_embeddings(texts)

    # Check the dimension of the embeddings
    embedding_dim = embeddings[0].shape[0] if embeddings else 0
    print(f"Embedding dimension: {embedding_dim}")

    # Create index with the correct number of dimensions
    create_index(index_name=index_name, dims=embedding_dim)

    documents = []
    for i, text in enumerate(texts):
        document = {
            "_id": str(uuid.uuid4()),  # Dynamic unique ID
            "text": text,
            "embedding": embeddings[i],
            "pdf_name": pdf_name
        }
        documents.append(document)

    # Bulk index the documents
    try:
        helpers.bulk(es, documents, index=index_name)
        print(f"Data from '{pdf_name}' added successfully to course '{course_name}'.")
    except BulkIndexError as e:
        for error in e.errors:
            print(f"Error indexing document: {error}")
        raise

def add_data_from_pdf(pdf_binary: bytes, course_name: str, pdf_name: str):
    """
    Add texts from a pdf, with the corresponding embedding, to an elasticsearch.

    Parameters:
    pdf_binary (bytes): The binary content of a PDF file.
    course_name (str): The name of the course in which we add data.
    pdf_name (str): The name of the PDF from which the chunks originate.
    """
    texts = chunk_text_from_pdf(pdf_binary=pdf_binary)
    add_data(texts, course_name=course_name, pdf_name=pdf_name)

def add_data_from_pdf_path(pdf_path: str, course_name: str, pdf_name: str):
    """
    Extract text from a PDF using its path, chunk it, and add to Elasticsearch.

    Parameters:
    pdf_path (str): Path to the PDF file.
    course_name (str): Course name for indexing.
    pdf_name (str): The name of the PDF.
    """
    pdf_binary = read_pdf_file_as_binary(pdf_path)
    chunks = chunk_text_from_pdf(pdf_binary, chunk_size_min=450, chunk_size_max=500)
    add_data(texts=chunks, course_name=course_name, pdf_name=pdf_name)

def get_relevant_chunks(text: str, index_name: str, top_n: int = 5):
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

    # Query Elasticsearch for documents with vectors
    response = es.search(index=index_name, body={
        "query": {
            "knn": {
                "embedding": {
                    "vector": input_embedding.tolist(),  # Convert numpy array to list
                    "k": top_n
                }
            }
        }
    })

    # Extract relevant chunks based on cosine similarity score
    relevant_chunks = []
    for doc in response['hits']['hits']:
        relevant_chunks.append(doc["_source"])

    return relevant_chunks

def remove_all_chunks_from_course(course_name: str):
    """
    Removes all chunks associated with a given course from Elasticsearch.

    Parameters:
    course_name (str): The name of the course in which we delete all chunks.
    """
    # Construct the index name based on exam and course
    index_name = f"course_{course_name}"

    # Check if index exists before trying to delete
    if es.indices.exists(index=index_name):
        # Delete index
        es.indices.delete(index=index_name)
        print(f"All chunks affiliated with course '{course_name}' have been removed successfully.")
    else:
        print(f"No index found for course '{course_name}'. Nothing to remove.")

def remove_chunks_by_pdf(course_name: str, pdf_name: str):
    """
    Removes all chunks associated with a specific PDF from Elasticsearch.

    Parameters:
    course_name (str): The name of the course associated with the chunks.
    pdf_name (str): The name of the PDF whose chunks need to be removed.
    """
    index_name = f"course_{course_name}"

    # Check if the index exists
    if not es.indices.exists(index=index_name):
        print(f"No index found for course '{course_name}'. Nothing to remove.")
        return

    # Delete documents with the specified pdf_name
    query = {
        "query": {
            "term": {
                "pdf_name": pdf_name  # Filter by PDF name
            }
        }
    }

    response = es.delete_by_query(index=index_name, body=query)
    deleted = response.get('deleted', 0)
    print(f"Deleted {deleted} chunks associated with '{pdf_name}' from course '{course_name}'.")

def get_all_pdf_names(course_name: str):
    """
    Retrieves all unique PDF names stored in Elasticsearch for a specified course.

    Parameters:
    course_name (str): The name of the course for which we search pdf names.

    Returns:
    set: A set of all unique PDF names.
    """
    pdf_names = set()

    index_name = f"course_{course_name}"

    # Check if the index exists
    if not es.indices.exists(index=index_name):
        print(f"No index found for course '{course_name}'. Nothing to retrieve.")
        return pdf_names

    # Query for unique pdf_names in the index
    query = {
        "aggs": {
            "unique_pdf_names": {
                "terms": {
                    "field": "pdf_name",
                    "size": 10000  # Adjust size as needed
                }
            }
        }
    }
    response = es.search(index=index_name, body=query)

    # Extract unique pdf_names from the response
    for bucket in response['aggregations']['unique_pdf_names']['buckets']:
        pdf_names.add(bucket['key'])

    return pdf_names
