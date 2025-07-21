# RAG WhatsApp & Web Chatbot for Academic Customer Service - Version 1.4.3

## Description

This project is an AI-powered chatbot system designed for the Program Studi Teknik Informatika at UIN Syarif Hidayatullah Jakarta. It provides academic customer service via WhatsApp (using both Twilio and WAHA), and a web interface. The chatbot leverages Retrieval-Augmented Generation (RAG) with document ingestion (PDF, TXT, CSV), vector search (FAISS), and advanced LLMs (Google Gemini via LangChain). An admin dashboard allows for document management, embedding, and monitoring.

## Features

- 🤖 **Dual WhatsApp Integration**: Twilio and WAHA (WhatsApp HTTP API)
- 💬 **Web Chat Interface**: Modern, responsive chat UI with Tailwind CSS
- 🗂️ **Admin Dashboard**: File upload, embedding, and knowledge base management
- 🔍 **Advanced RAG Pipeline**: Hybrid retrieval with semantic and keyword search
- 📄 **Document Ingestion**: PDF, TXT, CSV with chunking preview
- 🧠 **Google Gemini LLM** and **Nomic Atlas embeddings**
- 🏷️ **User Authentication**: Admin login with session management
- 📦 **Environment-based Configuration**: Flexible setup for different deployments
- 🩺 **Health and Test Endpoints**: System monitoring and diagnostics
- 🎨 **Production-ready Tailwind CSS**: Local build, no CDN dependencies
- 🔄 **Real-time File Watching**: Automatic knowledge base invalidation
- 📊 **Progress Tracking**: Embedding progress with real-time updates
- 🎯 **Jina AI Reranking**: Optional document reranking for better relevance

## Project Structure

```
Chatbot AI All Using API Ver 1.4.3/
│
├── app/                  # Core backend application code
│   ├── __init__.py       # Package initialization and Flask extensions
│   ├── core.py           # Main RAG logic, chat handling, embedding, and system info
│   ├── models.py         # Database models, LLM and embedding loader
│   └── vector_store.py   # Vector store (FAISS) creation, loading, and hybrid retrieval
│
├── main_whatsapp.py      # Main Flask app: WhatsApp, web chat, admin dashboard, API endpoints
├── requirements.txt      # Python dependencies
├── env.example           # Example environment variables
├── README.md             # Project documentation
│
├── documents/            # Folder for user-uploaded PDF, TXT, CSV files (knowledge base)
├── vector_db/            # FAISS vector store files (auto-generated)
├── instance/
│   └── app.db            # SQLite database (auto-generated)
│
├── templates/            # HTML templates for web chat and admin dashboard
│   ├── chat.html         # Web chat UI with real-time messaging
│   ├── admin_dashboard.html # Admin dashboard UI with file management
│   └── admin_login.html  # Admin login page
│
├── static/
│   └── css/
│       └── output.css    # Production Tailwind CSS (auto-generated)
│
├── src/
│   └── input.css         # Tailwind CSS entry point
│
├── tailwind.config.js    # Tailwind CSS configuration
├── package.json          # Node.js/Tailwind build config
├── package-lock.json     # Node.js lockfile
├── build_css.py          # Python script to build Tailwind CSS
│
├── functions/            # (Reserved for custom functions/scripts)
├── myenv/                # Python virtual environment (not tracked in git)
│
└── ...                   # Other config/cache folders (.gitignore, .mypy_cache, etc.)
```

**Key Components:**
- **app/**: All core backend logic, including RAG pipeline, document processing, and database models.
- **main_whatsapp.py**: Flask app entry point. Handles WhatsApp and web chat endpoints, admin dashboard, and API routes.
- **documents/**: Place your academic documents here for ingestion and embedding.
- **vector_db/**: Stores the FAISS vector index for fast semantic search (auto-generated).
- **instance/app.db**: SQLite database for admin users and file tracking (auto-generated).
- **templates/**: HTML templates for the web chat and admin dashboard.
- **static/css/output.css**: Production-ready Tailwind CSS (auto-generated, do not edit by hand).
- **src/input.css**: Tailwind CSS entry point (edit this to add custom CSS or Tailwind directives).
- **tailwind.config.js**: Tailwind CSS configuration (controls which files are scanned for classes).
- **package.json**: Node.js config for Tailwind build scripts.
- **build_css.py**: Python script to build Tailwind CSS (alternative to npm scripts).
- **functions/**: (Optional) For custom scripts or extensions.
- **myenv/**: Your Python virtual environment (should not be committed to version control).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Chatbot AI All Using API Ver 1.4.3"
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Copy `env.example` to `.env` and fill in your API keys and credentials:
     ```bash
     cp env.example .env
     ```
   - Edit `.env` with your API keys:

   **Required Environment Variables:**
   | Variable               | Description                                | Required For           |
   |------------------------|---------------------------------------------|-----------------------|
   | `GOOGLE_API_KEY`       | Google Gemini LLM API key                   | LLM (Gemini)          |
   | `NOMIC_API_KEY`        | Nomic Atlas API key for embeddings          | Embedding (Nomic)     |

   **Optional Environment Variables:**
   | Variable               | Description                                | Required For           |
   |------------------------|---------------------------------------------|-----------------------|
   | `JINA_API_KEY`         | Jina AI API key for reranking               | Rerank (Jina, optional)|
   | `SECRET_KEY`           | Flask secret key (auto-generated if not set)| Session security       |
   | `DATABASE_URL`         | Database URL (defaults to SQLite)           | Database               |
   | `PORT`                 | Server port (defaults to 5000)              | Server configuration   |

   > **Note**: Only `GOOGLE_API_KEY` and `NOMIC_API_KEY` are strictly required for basic operation. Add `JINA_API_KEY` for advanced document reranking if desired.

5. **Prepare documents:**
   - Place your PDF, TXT, or CSV files in the `documents/` folder.

6. **Initialize the database and admin user:**
   ```bash
   flask --app main_whatsapp.py init-db
   ```

7. **Install Node.js and npm (for Tailwind CSS build):**
   - Download and install from https://nodejs.org/

8. **Install Tailwind CSS dependencies:**
   ```bash
   npm install
   ```

9. **Build Tailwind CSS for production:**
   - **Option 1: Using npm script**
     ```bash
     npm run build-prod
     ```
   - **Option 2: Using Python script**
     ```bash
     python build_css.py
     ```
   - This will generate/update `static/css/output.css` for use in your templates.

10. **Run the application:**
    ```bash
    python main_whatsapp.py
    ```

## Usage

### Web Chat Interface
- **URL:** [http://localhost:5000/chat](http://localhost:5000/chat)
- **Features:** Real-time messaging, markdown support, conversation history

### WhatsApp Integration

#### Option 1: Twilio WhatsApp
- **Endpoint:** `/whatsapp` or `/webhook`
- **Configuration:** Point your Twilio WhatsApp webhook to your server's `/whatsapp` endpoint
- **Commands:** `/info`, `/help`, `/status`

#### Option 2: WAHA (WhatsApp HTTP API)
- **Endpoint:** `/api/whatsapp/webhook`
- **Configuration:** Configure WAHA to send webhooks to this endpoint
- **Features:** Direct WhatsApp integration without Twilio

### Admin Dashboard
- **URL:** [http://localhost:5000/admin](http://localhost:5000/admin)
- **Features:**
  - File upload with chunking preview
  - Knowledge base management
  - Real-time embedding progress
  - File status monitoring
  - Document deletion

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with system information |
| `/chat` | GET/POST | Web chat interface |
| `/admin` | GET/POST | Admin dashboard |
| `/admin/login` | GET/POST | Admin login |
| `/api/chat` | POST | Chat API for external integration |
| `/api/files` | GET | Get knowledge base files |
| `/api/kb_status` | GET | Knowledge base status |
| `/api/v1/preview-chunking` | POST | Preview document chunking |
| `/health` | GET | Health check endpoint |
| `/test` | GET | System test endpoint |

### Special Commands (WhatsApp)
- `/info` or `info` - System information and document status
- `/help` or `help` - Help and available services
- `/status` or `status` - System status

## Features in Detail

### RAG Pipeline
- **Document Processing:** PDF, TXT, CSV with intelligent chunking
- **Embedding:** Nomic Atlas embeddings for semantic search
- **Retrieval:** Hybrid approach combining semantic and keyword search
- **Reranking:** Optional Jina AI reranking for improved relevance
- **LLM:** Google Gemini 2.0 Flash for response generation

### Admin Dashboard Features
- **File Upload:** Drag-and-drop with chunking preview
- **Progress Tracking:** Real-time embedding progress with progress bars
- **File Management:** View, delete, and monitor knowledge base files
- **Status Monitoring:** Knowledge base health and embedding status
- **Responsive Design:** Works on desktop and mobile devices

### Security Features
- **Admin Authentication:** Secure login system
- **Session Management:** Flask-Login integration
- **File Validation:** Secure file upload with type checking
- **Environment Variables:** Secure configuration management

### Performance Features
- **File Watching:** Automatic knowledge base invalidation on file changes
- **Caching:** Efficient vector store loading and caching
- **Background Processing:** Non-blocking embedding operations
- **Optimized CSS:** Minified Tailwind CSS for production

## Tailwind CSS Setup (Production)

- **No CDN is used in production.**
- All templates reference `/static/css/output.css` for styling.
- To customize styles, edit `src/input.css` and rebuild.
- The build process scans all HTML templates and static JS/CSS for Tailwind classes (see `tailwind.config.js`).
- You can use either the npm scripts or the provided `build_css.py` Python script to build the CSS.

## Troubleshooting

### Common Issues

1. **Tailwind CSS not building:**
   - Ensure Node.js and npm are installed
   - Run `npm install` to install dependencies
   - Use `python build_css.py` as an alternative

2. **Embedding not working:**
   - Check that `NOMIC_API_KEY` is set in your `.env` file
   - Ensure documents are in the `documents/` folder
   - Check the admin dashboard for error messages

3. **WhatsApp integration issues:**
   - For Twilio: Verify webhook URL points to `/whatsapp`
   - For WAHA: Ensure webhook URL points to `/api/whatsapp/webhook`
   - Check server logs for detailed error messages

4. **Database issues:**
   - Run `flask --app main_whatsapp.py init-db` to recreate the database
   - Ensure the `instance/` directory is writable

### Logs and Debugging
- Check console output for detailed error messages
- Use `/health` endpoint to verify system status
- Use `/test` endpoint to test system functionality
- Monitor admin dashboard for embedding progress and errors

---

**Version 1.4.3 - Production-ready with dual WhatsApp integration, advanced RAG pipeline, and comprehensive admin dashboard!**


