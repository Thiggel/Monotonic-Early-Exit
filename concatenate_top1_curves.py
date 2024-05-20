from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2.generic import AnnotationBuilder
from PyPDF2._page import PageObject

def concatenate_pdfs(pdf_files, output_file):
    pdf_writer = PdfWriter()

    # Create a blank image with the width of three times the original page width
    width = 0
    height = 0
    pages = [None for _ in range(len(pdf_files))]
    pdf_readers = [None for _ in range(len(pdf_files))]

    for idx, pdf_file in enumerate(pdf_files):
        pdf_readers[idx] = PdfReader(pdf_file)
        pages[idx] = pdf_readers[idx].pages[0]  # Get the first page of each PDF

        width += pages[idx].mediabox.width
        height = max(height, pages[idx].mediabox.height)

    base_page = PageObject.create_blank_page(
        width=width,
        height=height
    )

    for page_idx, page in enumerate(pages):

        new_page = PageObject.create_blank_page(
            width=width,
            height=height
        )
        x_offset = float(page.mediabox.width * page_idx)
        translation = Transformation().translate(tx=x_offset, ty=0)
        new_page.merge_page(page)
        new_page.add_transformation(translation)
        base_page.merge_page(new_page)

    pdf_writer.add_page(base_page)

    # save the final pdf
    with open(output_file, 'wb') as out:
        pdf_writer.write(out)


pdf_files = [
    'top1_curves_vanilla.pdf',
    'top1_curves_calm.pdf',
    'top1_curves_free.pdf',
]

output_file = 'top1_curves_concatenated.pdf'

concatenate_pdfs(pdf_files, output_file)
