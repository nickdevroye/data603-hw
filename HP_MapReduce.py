"""
DATA603 – HW2: MapReduce (Harry Potter)
DOB: November 11, 2002

Steps:
  1) DOB => Book 6 (Half-Blood Prince), pages 11–20 and 102–111
  2) Extract those pages into file1.txt and file2.txt
  3) MapReduce word count on file1
  4) pyspellchecker to detect non-English tokens in file2
  5) Print results and save CSVs (for LaTeX inclusion)
"""

import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from spellchecker import SpellChecker
import PyPDF2
import math

#  Configuration 
# Path to Half-Blood Prince PDF
PDF_PATH = "half-blood-prince.pdf"
# Offset between printed pages and PDF internal index
OFFSET = 0
# Output folder
OUTDIR = "./out"

#  DOB => Book and Page Ranges 
BIRTH_MONTH = 11
BIRTH_DAY = 11
BIRTH_YEAR = 2002

BOOK_NUMBER = math.ceil(BIRTH_MONTH / 2) if BIRTH_MONTH >= 8 else BIRTH_MONTH  # => 6
FILE1_START = BIRTH_DAY                 # => page 11
FILE2_START = int("1" + str(BIRTH_YEAR)[-2:])  # => page 102
SPAN = 10                               # always 10 pages

#  PDF Extraction 
def extract_pages_to_text(pdf_path, start_printed, span, offset):
    reader = PyPDF2.PdfReader(pdf_path)
    texts = []
    for p in range(start_printed, start_printed + span):
        idx = (p - 1) + offset
        page = reader.pages[idx]
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

#  MapReduce Word Count 
WORD_RE = re.compile(r"[A-Za-z']+")

def tokenize(text):
    return [w for w in WORD_RE.findall(text)]

def mapper(lines):
    out = []
    for line in lines:
        for tok in tokenize(line):
            w = tok.lower().strip("'")
            if w and w not in ("'", "’"):
                out.append((w, 1))
    return out

def chunkify(lst, n):
    k = max(1, len(lst) // n) if lst else 1
    for i in range(0, len(lst), k):
        yield lst[i:i+k]

def shuffle(mapped_items):
    grouped = defaultdict(int)
    for k, v in mapped_items:
        grouped[k] += v
    return grouped

def mapreduce_wordcount_text(text, processes=None) -> Counter:
    lines = text.splitlines()
    nprocs = processes or max(1, min(cpu_count(), 8))
    chunks = list(chunkify(lines, nprocs))
    with Pool(processes=nprocs) as pool:
        mapped_lists = pool.map(mapper, chunks)
    combined = []
    for lst in mapped_lists:
        combined.extend(lst)
    grouped = shuffle(combined)
    return Counter(grouped)

#  Preprocess for Spellchecker 
def preprocess_for_spellcheck(s):
    s = s.replace("’", "'").replace("‘", "'").replace("—", "-").replace("–", "-")
    s = re.sub(r"[^A-Za-z']+", " ", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    tokens = s.split()
    cleaned = [t for t in tokens if not (len(t) >= 3 and t.isupper())]
    return " ".join(cleaned)

#  Non-English Detection 
spell = SpellChecker()

def detect_non_english_with_spellchecker(text, min_count=2) -> Counter:
    wc = mapreduce_wordcount_text(text)
    counts = Counter()
    for tok, c in wc.items():
        if c >= min_count and tok not in spell:
            counts[tok] = c
    return counts

#  Main 
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[DOB] 11/11/2002 => Book {BOOK_NUMBER} (Half-Blood Prince)")
    print(f"[Pages] file1: {FILE1_START}-{FILE1_START+SPAN-1} | file2: {FILE2_START}-{FILE2_START+SPAN-1}")

    # Extract pages and save file1.txt + file2.txt
    text1 = extract_pages_to_text(PDF_PATH, FILE1_START, SPAN, OFFSET)
    text2_raw = extract_pages_to_text(PDF_PATH, FILE2_START, SPAN, OFFSET)

    file1_path = os.path.join(OUTDIR, "file1.txt")
    file2_path = os.path.join(OUTDIR, "file2.txt")
    with open(file1_path, "w", encoding="utf-8") as f:
        f.write(text1)
    with open(file2_path, "w", encoding="utf-8") as f:
        f.write(text2_raw)

    # MapReduce wordcount on file1
    wc1 = mapreduce_wordcount_text(text1)
    top_words = wc1.most_common(40)
    all_words = wc1.most_common()

    # Non-English tokens on file2
    text2 = preprocess_for_spellcheck(text2_raw)
    noneng = detect_non_english_with_spellchecker(text2, min_count=2)

    # Print results
    print("\n=== Word Count (Top 40) — file1.txt ===")
    for w, c in top_words:
        print(f"{w:20s} {c:5d}")

    print("\n=== Non-English Tokens (pyspellchecker) — file2.txt ===")
    for t, c in noneng.most_common():
        print(f"{t:20s} {c:5d}")

    # Save CSVs for LaTeX
    wc_csv = os.path.join(OUTDIR, "file1_wordcount_top40.csv")
    wc_csv_all = os.path.join(OUTDIR, "file1_wordcount_all.csv")
    ne_csv = os.path.join(OUTDIR, "file2_nonenglish_pyspellchecker.csv")
    with open(wc_csv, "w", encoding="utf-8") as f:
        f.write("word,count\n")
        for w, c in top_words:
            f.write(f"{w},{c}\n")
    with open(wc_csv_all, "w", encoding="utf-8") as f:
        f.write("word,count\n")
        for w, c in all_words:
            f.write(f"{w},{c}\n")
    with open(ne_csv, "w", encoding="utf-8") as f:
        f.write("token,count\n")
        for t, c in noneng.most_common():
            f.write(f"{t},{c}\n")

if __name__ == "__main__":
    main()
