import io
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import Levenshtein
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import concurrent.futures
import nltk
import streamlit as st

# Download resources nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

# Cache untuk PDF dan hasil praproses
pdf_cache = OrderedDict()
preprocess_cache = OrderedDict()
CACHE_SIZE = 30
#Caching
def add_to_cache(url, text):
    if len(pdf_cache) >= CACHE_SIZE:
        pdf_cache.popitem(last=False)
    pdf_cache[url] = text
#PDFProcessing
def extract_text_from_pdf(pdf_file, method="pdfplumber"):
    try:
        if method == "pdfplumber":
            with pdfplumber.open(pdf_file) as pdf:
                return "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif method == "PyPDF2":
            reader = PyPDF2.PdfReader(pdf_file)
            return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif method == "PyMuPDF":
            pdf_document = fitz.open("pdf", pdf_file.read())
            return "".join([page.get_text() for page in pdf_document])
        else:
            return ""
    except Exception as e:
        st.error(f"Gagal mengekstrak teks dari PDF: {e}")
        return ""

preprocess_cache = {}

# Fungsi untuk case folding
def case_folding(text):
    return text.lower()  # Mengubah teks menjadi huruf kecil

# Fungsi untuk tokenizing
def tokenizing(text):
    return word_tokenize(text)  # Memecah teks menjadi kata-kata (token)

# Fungsi untuk menghapus stopwords
def remove_stopwords(words):
    stop_words = set(stopwords.words('indonesian'))  # Daftar stopwords bahasa Indonesia
    filtered_words = [word for word in words if word not in stop_words]  # Menghapus stopwords
    return filtered_words

# Fungsi untuk stemming
def stemming(words):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()  # Membuat objek stemmer
    stemmed_words = [stemmer.stem(word) for word in words]  # Melakukan stemming
    return stemmed_words

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    if text in preprocess_cache:
        return preprocess_cache[text]  # Mengembalikan hasil dari cache jika ada
    
    # Proses teks
    text = case_folding(text)  # Mengubah teks menjadi huruf kecil
    words = tokenizing(text)  # Tokenisasi
    words = remove_stopwords(words)  # Menghapus stopwords
    words = stemming(words)  # Stemming
    
    processed_text = " ".join(words)  # Menggabungkan kembali kata-kata yang telah diproses
    
    # Menyimpan hasil ke cache
    preprocess_cache[text] = processed_text
    return processed_text  # Mengembalikan teks yang telah diproses
from Levenshtein import ratio

def combined_similarity(text1, text2):
    # Hitung kemiripan menggunakan Levenshtein
    levenshtein_score = ratio(text1, text2)
    return levenshtein_score * 100  # Kembalikan dalam persen

def fetch_pdf_links():
    url = "https://flask-x9m7.onrender.com/get_pdf_links"  # Ubah ke URL API yang sesuai
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        pdf_links = data.get("pdf_links", [])
        if pdf_links:
            return pdf_links
        else:
            st.error("Tidak ada tautan PDF yang ditemukan pada respons.")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mendapatkan data dari {url}. Error: {e}")
        return []

pdf_cache = {}

def fetch_and_preprocess_text(url, method="pdfplumber"):
    if url in pdf_cache:
        return pdf_cache[url]  # Mengembalikan hasil dari cache jika ada

    try:
        response = requests.get(url, timeout=10)  # Mengambil konten dari URL
        response.raise_for_status()  # Memeriksa apakah permintaan berhasil
        
        pdf_file = io.BytesIO(response.content)  # Membaca konten menjadi file PDF dalam format bytes
        
        # Ekstrak teks dari PDF berdasarkan metode yang dipilih
        if method == "pdfplumber":
            with pdfplumber.open(pdf_file) as pdf:  # Menggunakan pdfplumber.open
                text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif method == "PyPDF2":
            reader = PyPDF2.PdfReader(pdf_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif method == "PyMuPDF":
            pdf_document = fitz.open("pdf", pdf_file.read())
            text = "".join([page.get_text() for page in pdf_document])
        else:
            st.error("Metode ekstraksi tidak valid.")  # Menampilkan kesalahan jika metode tidak valid
            return None
        
        processed_text = preprocess_text(text)  # Memproses teks yang diekstrak
        
        add_to_cache(url, processed_text)  # Menyimpan hasil ke cache
        return processed_text  # Mengembalikan teks yang telah diproses
    
    except Exception as e:
        st.error(f"Gagal memproses URL {url}. Error: {e}")  # Menampilkan pesan kesalahan
        return None  # Mengembalikan None jika terjadi kesalahan

def check_similarity(uploaded_text, pdf_links, method="pdfplumber"):
    max_similarity, best_url = 0, ""  # Inisialisasi nilai maksimum kesamaan dan URL terbaik
    
    with ProcessPoolExecutor() as executor:
        # Menjalankan fetch_and_preprocess_text untuk setiap URL dalam pdf_links secara paralel
        future_to_url = {executor.submit(fetch_and_preprocess_text, url, method): url for url in pdf_links}
        
        # Mengumpulkan hasil dari setiap future yang telah selesai
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]  # Mendapatkan URL terkait dari future
            processed_text = future.result()  # Mengambil hasil dari future
            
            if processed_text is not None:  # Pastikan hasil tidak None
                # Menghitung kesamaan menggunakan combined_similarity
                similarity = combined_similarity(uploaded_text, processed_text)
                
                # Memperbarui nilai maksimum kesamaan dan URL terbaik jika kesamaan lebih tinggi
                if similarity > max_similarity:
                    max_similarity, best_url = similarity, url
    
    return max_similarity, best_url  # Mengembalikan kesamaan maksimum dan URL terbaik


# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Deteksi Plagiarisme")

    # Upload PDF
    uploaded_pdf = st.file_uploader("Unggah file PDF", type="pdf")

    if uploaded_pdf:
        # Pilihan metode ekstraksi
        extraction_method = st.selectbox(
            "Pilih metode ekstraksi teks PDF:", 
            ("PyPDF2", "pdfplumber", "PyMuPDF")
        )

        # Ekstraksi teks PDF sesuai pilihan metode
        pdf_text = extract_text_from_pdf(uploaded_pdf, extraction_method)

        # Tombol untuk memulai cek plagiarisme
        if st.button("Cek Plagiarisme"):
            start_time = time.time()

            # Preprocessing teks PDF
            processed_pdf_text = preprocess_text(pdf_text)

            # Ambil daftar link PDF dari JSON via API
            pdf_links = fetch_pdf_links()

            if processed_pdf_text and pdf_links:
                plagiarism_percentage, best_match_url = check_similarity(
                    processed_pdf_text, pdf_links, extraction_method
                )
                accuracy = 100 - plagiarism_percentage

                end_time = time.time()
                processing_time = end_time - start_time

                st.success(f"Persentase plagiarisme : {plagiarism_percentage:.2f}% ")
                st.info(f"Akurasi deteksi: {accuracy:.2f}%")
                st.write(f"Lama proses: {processing_time:.2f} detik")
                if best_match_url:
                    st.write(f"Tautan dengan kemiripan tertinggi: {best_match_url}")
            else:
                st.error("Gagal memproses teks PDF atau mendapatkan data pembanding.")
    else:
        st.info("Tunggu tombol ditekan untuk memulai cek plagiarisme.")

if __name__ == "__main__":
    main()
#>>>>>>> c627ed8 (Add start.sh)
