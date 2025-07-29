# 🎓 Customer Service Teknik Informatika UIN Jakarta

Chatbot AI Customer Service untuk Program Studi Teknik Informatika UIN Syarif Hidayatullah Jakarta.

## 🌟 Fitur Utama

- **RAG (Retrieval-Augmented Generation)**: Menggunakan dokumen akademik untuk jawaban yang akurat
- **Multi-Platform**: WhatsApp (Twilio), Web Chat, dan API
- **Knowledge Base Management**: Upload dan kelola dokumen PDF, TXT, CSV
- **Fitur "Jelaskan Lebih Jelas"**: Dapatkan penjelasan lebih detail dari jawaban sebelumnya
- **Fitur "Pertanyaan Umum" (FAQ)**: 30+ FAQ dengan jawaban instan
- **Fitur FAQ WhatsApp**: Menu FAQ interaktif untuk platform text-based
- **LangSmith Integration**: Comprehensive LLM observability dan tracing

## 🚀 LangSmith Integration

Proyek ini telah terintegrasi dengan **LangSmith** untuk comprehensive tracing, monitoring, dan performance tracking dari LLM operations.

### 🔧 Setup LangSmith

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Buat file `.env` berdasarkan `env.example` dan tambahkan:
   ```env
   # LangSmith Configuration for LLM Observability
   LANGCHAIN_TRACING_V2="true"
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
   LANGCHAIN_PROJECT="UIN Jakarta TI Chatbot"
   ```

3. **Get LangSmith API Key**:
   - Daftar di [LangSmith](https://smith.langchain.com/)
   - Buat API key di dashboard
   - Masukkan ke `LANGCHAIN_API_KEY`

### 📊 What's Being Traced

- **All LLM Calls**: Setiap panggilan ke Google Gemini 2.0 Flash
- **RAG Chain Operations**: Document retrieval, embedding, dan response generation
- **User Interactions**: Metadata lengkap untuk setiap request
- **Performance Metrics**: Latency, token usage, dan error tracking
- **Platform-Specific Data**: WhatsApp, Web, dan API interactions

### 🔍 Tracing Features

- **Automatic Tracing**: LangChain otomatis trace semua LLM dan chain operations
- **Manual Metadata**: User ID, platform, conversation state, dan custom metadata
- **Error Tracking**: Comprehensive error logging dan debugging
- **Performance Monitoring**: Real-time latency dan throughput metrics

### 📈 Dashboard Access

Setelah setup, akses dashboard LangSmith untuk melihat:
- **Project Overview**: Summary metrics dan recent runs
- **Detailed Traces**: Step-by-step execution flow
- **Performance Analytics**: Latency trends dan optimization opportunities
- **Error Analysis**: Failed requests dan debugging information

## 🛠️ Installation & Setup

### Prasyarat
- **Python 3.8+**
- **Node.js 16+** (untuk Tailwind CSS)
- **API Keys**: Google Gemini, Nomic Atlas (wajib), Jina AI (opsional)

### Setup Langkah demi Langkah

1. **Clone dan navigasi ke proyek:**
   ```bash
   git clone <your-repo-url>
   cd "Chatbot AI All Using API Ver 1.4.5"
   ```

2. **Buat dan aktifkan lingkungan virtual:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Di Windows: myenv\Scripts\activate
   ```

3. **Instal dependensi Python:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi variabel lingkungan:**
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` dengan API key Anda:
   ```env
   # Wajib
   GOOGLE_API_KEY="your_google_gemini_api_key"
   NOMIC_API_KEY="your_nomic_atlas_api_key"
   
   # Opsional
   JINA_API_KEY="your_jina_rerank_api_key"
   SECRET_KEY="your_flask_secret_key"
   DATABASE_URL="sqlite:///app.db"
   PORT=5000
   ```

5. **Inisialisasi database dan buat user admin:**
   ```bash
   flask --app main_whatsapp.py init-db
   ```

6. **Instal dependensi Tailwind CSS:**
   ```bash
   npm install
   ```

7. **Build CSS produksi:**
   ```bash
   # Opsi 1: Menggunakan npm
   npm run build-prod
   
   # Opsi 2: Menggunakan script Python
   python build_css.py
   ```

8. **Tambahkan dokumen akademik Anda:**
   - Letakkan file PDF, TXT, atau CSV di folder `documents/`
   - Format yang didukung: Panduan akademik, katalog mata kuliah, daftar dosen, dll.

9. **Jalankan aplikasi:**
   ```bash
   python main_whatsapp.py
   ```

## 🎯 Penggunaan

### Antarmuka Web Chat
- **URL:** `http://localhost:5000/chat`
- **Fitur:** Pesan real-time, dukungan markdown, riwayat percakapan
- **Bahasa:** Indonesia (dioptimalkan untuk konteks akademik)

### Integrasi WhatsApp

#### Twilio WhatsApp
- **Endpoint:** `/whatsapp` atau `/webhook`
- **Setup:** Konfigurasi URL webhook Twilio ke server Anda


#### WAHA (WhatsApp HTTP API)
- **Endpoint:** `/api/whatsapp/webhook`
- **Setup:** Konfigurasi WAHA untuk mengirim webhook ke endpoint ini
- **Fitur:** Integrasi WhatsApp langsung tanpa Twilio

### Dashboard Admin
- **URL:** `http://localhost:5000/admin`
- **Fitur:**
  - Upload file dengan preview chunking
  - Kemajuan embedding real-time
  - Pengelolaan basis pengetahuan
  - Penghapusan dokumen dan manajemen vector store

### Endpoint API

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/` | GET | Halaman beranda dengan informasi sistem |
| `/chat` | GET/POST | Antarmuka web chat |
| `/admin` | GET/POST | Dashboard admin |
| `/admin/login` | GET/POST | Autentikasi admin |
| `/api/chat` | POST | API chat untuk integrasi eksternal |
| `/api/files` | GET | File basis pengetahuan |
| `/api/kb_status` | GET | Status basis pengetahuan |
| `/api/v1/preview-chunking` | POST | Preview chunking dokumen |
| `/whatsapp` | POST | Webhook WhatsApp Twilio |
| `/api/whatsapp/webhook` | POST | Webhook WhatsApp WAHA |
| `/health` | GET | Health check |
| `/test` | GET | Test sistem |

### Perintah Khusus
- `Jelaskan Lebih Jelas` - Dapatkan penjelasan lebih detail dari respons sebelumnya

## 🔧 Fitur Canggih

### Detail Pipeline RAG
- **Pemrosesan Dokumen:** Chunking cerdas dengan ukuran/overlap yang dapat dikonfigurasi
- **Retrieval Hibrida:** Menggabungkan pencarian semantik dengan pencocokan kata kunci
- **Reranking:** Reranking Jina AI opsional untuk relevansi yang lebih baik
- **Memori Konteks:** Melacak konteks percakapan untuk pertanyaan lanjutan

### Fitur "Jelaskan Lebih Jelas"
- **Trigger Detection:** Mendeteksi frasa seperti "Jelaskan lebih jelas", "Explain more", "Tell me more"
- **State Management:** Menyimpan respons bot terakhir untuk setiap pengguna
- **Elaboration AI:** Menggunakan prompt khusus untuk memberikan penjelasan yang lebih detail
- **UI Integration:** Tombol "Jelaskan Lebih Jelas" di setiap respons bot
- **Multi-language Support:** Mendukung trigger dalam Bahasa Indonesia dan Inggris

### Fitur "Pertanyaan Umum" (FAQ)

- **Client-side FAQ**: Pertanyaan dan jawaban disimpan dalam `faq_data.json`
- **Interactive UI**: Tombol klik untuk pertanyaan umum
- **Instant Responses**: Jawaban langsung tanpa API call
- **Toggle Control**: Tombol untuk menampilkan/menyembunyikan FAQ
- **Responsive Design**: Tampilan optimal untuk desktop dan mobile

### Fitur FAQ WhatsApp

- **Trigger Footer**: Setiap respons bot menampilkan "Ketik 'Menu FAQ' untuk daftar pertanyaan umum"
- **Menu FAQ Command**: Pengguna dapat mengetik "Menu FAQ" untuk melihat daftar pertanyaan
- **Numbered Selection**: Pengguna memilih pertanyaan dengan nomor (1, 2, 3, dst.)
- **State Management**: Sistem mengingat konteks FAQ untuk setiap pengguna
- **Error Handling**: Validasi input dan pesan error yang informatif
- **Auto-reset**: Konteks FAQ otomatis reset setelah pemilihan atau input tidak valid

### Fitur Dashboard Admin
- **Upload File:** Drag-and-drop dengan preview chunking
- **Pelacakan Kemajuan:** Kemajuan embedding real-time dengan indikator visual
- **Pengelolaan File:** Lihat, hapus, dan pantau file basis pengetahuan
- **Manajemen Vector Store:** Hapus dan rebuild database vektor
- **Desain Responsif:** Berfungsi di desktop dan mobile

### Fitur Keamanan
- **Autentikasi Admin:** Login aman dengan hashing password
- **Manajemen Sesi:** Integrasi Flask-Login
- **Validasi File:** Upload file aman dengan pengecekan tipe
- **Variabel Lingkungan:** Manajemen konfigurasi yang aman

### Fitur Performa
- **Pemrosesan Background:** Operasi embedding non-blocking
- **File Watching:** Invalidasi basis pengetahuan otomatis
- **Caching:** Loading dan caching vector store yang efisien
- **CSS Teroptimasi:** Tailwind CSS terkompresi untuk produksi

## 🎨 Setup Tailwind CSS

### Build Produksi
- **Tanpa ketergantungan CDN** - semua CSS dibangun secara lokal
- **Output terkompresi** untuk performa optimal
- **Konfigurasi kustom** di `tailwind.config.js`
- **Opsi build:**
  ```bash
  # Development (mode watch)
  npm run build
  
  # Production (terkompresi)
  npm run build-prod
  
  # Script Python alternatif
  python build_css.py
  ```

### Kustomisasi
- Edit `src/input.css` untuk menambah style kustom atau direktif Tailwind
- Modifikasi `tailwind.config.js` untuk mengontrol file mana yang di-scan
- CSS dibangun otomatis ketika template berubah

## 🔍 Troubleshooting

### Masalah Umum

1. **Tailwind CSS tidak terbangun:**
   ```bash
   # Pastikan Node.js terinstal
   node --version
   
   # Reinstal dependensi
   npm install
   
   # Gunakan alternatif Python
   python build_css.py
   ```

2. **Embedding tidak berfungsi:**
   - Periksa `NOMIC_API_KEY` di file `.env`
   - Pastikan dokumen ada di folder `documents/`
   - Periksa dashboard admin untuk pesan error
   - Verifikasi izin file

3. **Masalah integrasi WhatsApp:**
   - **Twilio:** Verifikasi URL webhook mengarah ke `/whatsapp`
   - **WAHA:** Pastikan URL webhook mengarah ke `/api/whatsapp/webhook`
   - Periksa log server untuk pesan error detail
   - Verifikasi API key dan konektivitas jaringan

4. **Masalah database:**
   ```bash
   # Buat ulang database
   flask --app main_whatsapp.py init-db
   
   # Periksa izin folder instance
   ls -la instance/
   ```

### Alat Debugging
- **Health check:** `GET /health`
- **Test sistem:** `GET /test`
- **Dashboard admin:** Pantau kemajuan embedding dan error
- **Log konsol:** Pesan error detail dan info debugging

### Optimasi Performa
- **Caching vector store:** Loading dan penggunaan memori yang efisien
- **Pemrosesan background:** Operasi non-blocking
- **File watching:** Update otomatis ketika dokumen berubah
- **CSS teroptimasi:** Stylesheet terkompresi dan terkompresi

## 📚 Konteks Akademik

Chatbot ini dirancang khusus untuk **Program Studi Teknik Informatika UIN Syarif Hidayatullah Jakarta** dan dapat menangani:

- **Informasi kurikulum** dan katalog mata kuliah
- **Kebijakan akademik** dan prosedur administrasi
- **Informasi dosen** dan jadwal mengajar
- **Panduan registrasi** dan prosedur pendaftaran
- **Panduan PKL (Praktik Kerja Lapangan)** dan skripsi
- **Fasilitas kampus** dan layanan mahasiswa

## 🔄 Riwayat Versi

### Versi 1.4.5 (Saat Ini)
- **Integrasi WhatsApp ganda** (Twilio + WAHA)
- **Pipeline RAG canggih** dengan retrieval hibrida
- **Dashboard admin real-time** dengan pelacakan kemajuan
- **Fungsionalitas preview chunking dokumen**
- **Optimasi bahasa Indonesia**
- **Tailwind CSS siap produksi**
- **Penanganan error komprehensif**

### Versi Sebelumnya
- **1.4.3**: Integrasi WhatsApp dasar dan pipeline RAG
- **1.4.0**: Rilis awal dengan fitur web chat dan admin

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur
3. Lakukan perubahan Anda
4. Test secara menyeluruh
5. Submit pull request

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file LICENSE untuk detail.

---

**Versi 1.4.5 - Chatbot akademik siap produksi dengan integrasi WhatsApp ganda, pipeline RAG canggih, dan dashboard admin komprehensif! 🎓🤖**


