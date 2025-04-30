import sys

from es import add_data_from_pdf_path

if len(sys.argv) != 4:
    print("Usage: python3 add_pdf_to_es.py <pdf_file_path> <course_name> <pdf_name>")
    sys.exit(1)

# Retrieve arguments
pdf_file_path = sys.argv[1]
course_name = sys.argv[2]
pdf_name = sys.argv[3]

# Call the existing function with provided arguments
add_data_from_pdf_path(pdf_path=pdf_file_path, course_name=course_name,pdf_name=pdf_name)

# print(f"PDF data added to Elasticsearch successfully from {pdf_file_path}")
# print(f"Pdf: {pdf_name} Course: {course_name}")
