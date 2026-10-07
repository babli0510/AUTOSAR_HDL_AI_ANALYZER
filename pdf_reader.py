import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing page number and extracted text.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "data/raw/sample_hld.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"Total pages: {len(pages)}")

    for page in pages[:2]:
        print("\n" + "=" * 60)
        print(f"PAGE {page['page_number']}")
        print("=" * 60)
        print(page["text"][:1000])