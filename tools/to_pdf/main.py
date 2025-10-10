import os
import re
from urllib.parse import urljoin, urlparse

import pdfkit
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfMerger, PdfReader, PdfWriter


def natural_key(s):
    """Turn a string into a list of ints and text, so 's10' > 's9'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def get_all_links(base_url):
    """Crawl all internal links from base_url."""
    visited = set()
    to_visit = [base_url]
    urls = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url)
            r.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        urls.append(url)

        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            # Only keep pages within same domain
            if urlparse(href).netloc == urlparse(base_url).netloc:
                if href not in visited and href not in to_visit and "#" not in href:
                    to_visit.append(href)
    return urls


def save_pages_as_pdfs(urls, output_dir="pages_pdfs"):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = []
    for i, url in enumerate(urls):
        out_file = os.path.join(output_dir, f"page_{i + 1}.pdf")
        try:
            pdfkit.from_url(url, out_file)
            pdf_files.append(out_file)
            print(f"✅ Saved {url} → {out_file}")
        except Exception as e:
            print(f"❌ Failed to render {url}: {e}")
    return pdf_files


def merge_pdfs(pdf_files, output_file="combined.pdf"):
    merger = PdfMerger()
    for pdf in pdf_files:
        merger.append(pdf)
    merger.write(output_file)
    merger.close()
    print(f"📚 Combined PDF saved as {output_file}")


import subprocess


def compress_pdf(input_file, output_file, quality="/ebook"):
    subprocess.run(
        [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={quality}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={output_file}",
            input_file,
        ]
    )
    print(f"📉 Compressed {input_file} → {output_file}")


def split_pdf(input_file, output_files):
    reader = PdfReader(input_file)
    total_pages = len(reader.pages)
    num_splits = len(output_files)
    pages_per_split = total_pages // num_splits
    remainder = total_pages % num_splits

    start = 0
    for i, out_file in enumerate(output_files):
        # Distribute the remainder pages one by one to the first splits
        end = start + pages_per_split + (1 if i < remainder else 0)
        writer = PdfWriter()
        for page in reader.pages[start:end]:
            writer.add_page(page)
        with open(out_file, "wb") as f:
            writer.write(f)
        print(f"✂️ Split pages {start + 1}-{end} → {out_file}")
        start = end


if __name__ == "__main__":
    base_url = "https://skaftenicki.github.io/dtu_mlops/"
    urls = get_all_links(base_url)
    urls = sorted(urls, key=natural_key)  # 👈 use natural sort
    print(f"Found {len(urls)} pages.")

    pdf_files = save_pages_as_pdfs(urls)
    merge_pdfs(pdf_files, "dtu_mlops_all.pdf")
    split_pdf(
        "dtu_mlops_all.pdf",
        ["dtu_mlops_part1.pdf", "dtu_mlops_part2.pdf", "dtu_mlops_part3.pdf", "dtu_mlops_part4.pdf"],
    )
    compress_pdf("dtu_mlops_part1.pdf", "dtu_mlops_part1_small.pdf")
    compress_pdf("dtu_mlops_part2.pdf", "dtu_mlops_part2_small.pdf")
    compress_pdf("dtu_mlops_part3.pdf", "dtu_mlops_part3_small.pdf")
    compress_pdf("dtu_mlops_part4.pdf", "dtu_mlops_part4_small.pdf")
