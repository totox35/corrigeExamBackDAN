import sys
from es import remove_all_chunks_from_course, remove_chunks_by_pdf

def delete_chunks(course: str, pdf_name: str = None):
    """
    Delete chunks related to course and specific pdf.

    Parameters:
    course (str): The name of the course in which we delete chunks.
    pdf_name (str, optional): The name of the PDF whose chunks need to be removed. If None, all chunks are deleted from the course.
    """
    if pdf_name:
        # Delete only chunks associated with the specific PDF
        remove_chunks_by_pdf(course_name=course, pdf_name=pdf_name)
    else:
        # Delete all chunks in the course
        remove_all_chunks_from_course(course_name=course)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 delete_chunks_from_es.py <course_name> [pdf_name]")
        sys.exit(1)

    course_name = sys.argv[1]
    pdf = sys.argv[2] if len(sys.argv) > 2 else None
    delete_chunks(course_name, pdf)
