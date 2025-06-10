import fitz  # PyMuPDF
import os

pdf_basenames = ['TESLA', 'WALMART', 'VISA']  # Add your PDF filenames here

BASE_INPUT_DIR = 'PDFs'
BASE_OUTPUT_DIR = 'Outputs'

print('Starting extraction...')

for pdf_name in pdf_basenames:
 
    pdf_path = f"{BASE_INPUT_DIR}/{pdf_name}.pdf"
    output_dir = f"{BASE_OUTPUT_DIR}/{pdf_name}"

    images_dir = f"{output_dir}/images"
    text_file_path = f"{output_dir}/text.txt"
    tables_file_path = f"{output_dir}/tables.txt"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_path}: {e}")
        continue

    print(f"Processing '{pdf_name}.pdf' with {len(doc)} pages...")

   
    with open(text_file_path, 'w', encoding='utf-8') as text_file, \
         open(tables_file_path, 'w', encoding='utf-8') as tables_file:
        
        for page_index in range(len(doc)):
            
            # get page no.
            page = doc[page_index]
            page_number = page_index + 1
            print(f"  - Page {page_number}/{len(doc)}")

            # Extract and save text
            full_text = page.get_text("text")
            text_file.write(f"--- Page {page_number} ---\n")
            text_file.write(full_text + "\n")

            # Extract and save tables in given page
            table_finder = page.find_tables()
            tables = table_finder.tables
            if tables:
                
                tables_file.write(f"--- Page {page_number} Tables ---\n")
                
                for table_index, table in enumerate(tables, start=1):
                    rows = table.extract()
                    tables_file.write(f"Table {table_index}:\n")
                    
                    for row in rows:
                        # Handle potential None values in table cells
                        tables_file.write("\t".join(cell if cell is not None else '' for cell in row) + "\n")
                    tables_file.write("\n")

            # Extract and save images
            image_list = page.get_images(full=True) 
            for img_info in image_list:
                
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
               
                img_filename = f"image{xref}-page{page_number}.png"
                img_path = f"{images_dir}/{img_filename}"
                
                if pix.n < 5:  # Not CMYK
                    pix.save(img_path)
                else:  # Convert CMYK to RGB
                    pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                    pix_rgb.save(img_path)
                    pix_rgb = None
                pix = None

    print(f"Finished '{pdf_name}'. Outputs are in '{output_dir}'.\n")

print('Extraction complete for all PDFs!')