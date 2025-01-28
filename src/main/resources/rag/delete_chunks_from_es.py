import sys
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def delete_chunks(course_name, pdf_name=None):
    index_name = course_name

    # Check if the index exists
    if not es.indices.exists(index=index_name):
        print(f"No index found for course '{course_name}'. Nothing to delete.")
        return

    if pdf_name:
        # Delete only chunks associated with the specific PDF
        query = {
            "query": {
                "term": {
                    "pdf_name": pdf_name
                }
            }
        }
        response = es.delete_by_query(index=index_name, body=query)
        print(f"Deleted {response.get('deleted', 0)} chunks from PDF '{pdf_name}' in course '{course_name}'.")
    else:
        # Delete all chunks in the course
        es.indices.delete(index=index_name)
        print(f"All chunks in course '{course_name}' deleted successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 delete_chunks_from_es.py <course_name> [pdf_name]")
        sys.exit(1)

    course_name = sys.argv[1]
    pdf_name = sys.argv[2] if len(sys.argv) > 2 else None
    delete_chunks(course_name, pdf_name)