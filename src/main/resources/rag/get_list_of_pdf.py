import sys
from es import get_all_pdf_names

if len(sys.argv) != 2:
    print("Usage: python3 get_list_of_pdf.py  <course_name>")
    sys.exit(1)

course_name = sys.argv[1]
pdf_names = get_all_pdf_names(course_name=course_name)
for pdf_name in pdf_names:
    print(pdf_name)
