import os
from pypdf import PdfReader

def read_all_pdfs(folder_path):
    """Opens every PDF in a folder and pulls out the text."""
    documents = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return documents

    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    if not files:
        print(f"No PDF files found in '{folder_path}'. Please add one!")
        return documents

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        print(f"Reading: {filename}")

        reader = PdfReader(filepath)
        text = ""
        for page_num, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        documents.append({
            "filename": filename,
            "text": text,
            "page_count": len(reader.pages)
        })

    print(f"\nDone! Successfully read {len(documents)} document(s).")
    return documents


if __name__ == "__main__":
    docs = read_all_pdfs("data/pdfs")

    if docs:
        print(f"\n--- Preview of '{docs[0]['filename']}' (Total Pages: {docs[0]['page_count']}) ---")
        # Print the first 500 characters
        print(docs[0]["text"][:500])