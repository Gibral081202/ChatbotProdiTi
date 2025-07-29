from typing import Dict, List, Optional, Any
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from .models import load_llm, load_embedding_model
from .vector_store import load_vector_store, hybrid_retrieve
import traceback
import re
import threading
import json
import os
from app.models import KnowledgeBaseFile, db
from app.vector_store import create_vector_store
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd  # type: ignore
import hashlib
from langsmith import traceable

# Store user sessions to track new users
user_sessions: Dict[str, bool] = {}

# Store last bot response for each user for "Explain More" feature
last_bot_responses: Dict[str, str] = {}

# Store user context for FAQ state management
user_faq_context: Dict[str, str] = {}

embedding_progress: Dict[str, Any] = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "current": 0,
    "message": "",
}


def format_links_for_chat(text: str) -> str:
    """
    Fix URL formatting in chatbot responses to make links clickable in chat interfaces.
    
    This function converts Markdown-formatted links in lists to raw clickable URLs.
    
    Args:
        text (str): The response text to format
        
    Returns:
        str: Text with properly formatted links
    """
    if not text:
        return text
    
    # Pattern to match Markdown list items with links: * [text](url)
    # This will match patterns like:
    # * [https://example.com](https://example.com)
    # * [Kurikulum](https://example.com)
    markdown_link_pattern = r'\* \[([^\]]+)\]\(([^)]+)\)'
    
    def replace_markdown_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        
        # If the link text is already a URL, just return the URL
        if link_text.startswith(('http://', 'https://')):
            return f"{link_url}"
        else:
            # If it's descriptive text, format as [text](url)
            return f"[{link_text}]({link_url})"
    
    # Replace all Markdown list links
    formatted_text = re.sub(markdown_link_pattern, replace_markdown_link, text)
    
    # Also handle cases where URLs are just listed without Markdown formatting
    # Pattern to match lines that start with * and contain URLs
    url_list_pattern = r'\* (https?://[^\s\n]+)'
    
    def replace_url_list(match):
        url = match.group(1)
        return f"{url}"
    
    formatted_text = re.sub(url_list_pattern, replace_url_list, formatted_text)
    
    # Clean up multiple consecutive newlines
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    return formatted_text


@traceable(
    run_type="chain", 
    name="RAG Chain Creation"
)
def create_rag_chain():
    """
    Create and return a RAG chain with LLM, embeddings, and vector store.
    
    Returns:
        Tuple: Configured RAG chain components
    """
    # Load models
    llm = load_llm()
    embeddings = load_embedding_model()
    
    # Load vector store
    vector_store = load_vector_store(embeddings)
    if not vector_store:
        raise ValueError("Moho maaf tolong perjelasan pertanyaan anada.")
    
    # Use a dummy retriever, we'll override retrieval below
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 1}
    )
    
    # SINGLE authoritative prompt for all responses
    prompt = ChatPromptTemplate.from_template(
        """Anda adalah asisten AI Help Desk profesional, ramah, dan ahli untuk Program Studi Teknik Informatika UIN Syarif Hidayatullah Jakarta. Tugas Anda adalah menjawab pertanyaan berdasarkan informasi yang diberikan dalam <context> berikut.

### ATURAN MUTLAK:
1. **Selalu jawab dalam Bahasa Indonesia yang baik, sopan, dan profesional.**
2. **Gunakan hanya informasi dari <context> untuk menjawab.**
3. **JANGAN PERNAH mengulang atau menulis ulang kalimat, poin, atau informasi apapun.**
4. **Gabungkan dan sintesis informasi dari <context> menjadi jawaban yang mengalir, jelas, dan mudah dipahami.**
5. **Gunakan format Markdown:**
   - Gunakan heading (###, ####) untuk judul dan subjudul.
   - Gunakan bullet list (*) dan numbered list (1.) sesuai kebutuhan.
   - Gunakan **bold** untuk kata kunci penting.
6. **Jika informasi tidak ditemukan di <context>, jawab dengan kalimat standar:**
   - "Tolong perjelas terkait pertanyaan yang Anda berikan."
7. **Akhiri setiap jawaban dengan pertanyaan ramah untuk mendorong interaksi lanjutan.**
8. **Jangan pernah menyalin mentah dari <context>; selalu parafrase dan rangkum.**

<context>
{documents}
</context>

**Pertanyaan Pengguna:**
{input}

"""
    )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(
        llm=llm, prompt=prompt, document_variable_name="documents"
    )
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=document_chain
    )
    
    return rag_chain, vector_store, embeddings, document_chain


def is_new_user(user_id: str) -> bool:
    """
    Check if user is new based on their ID.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        bool: True if user is new, False otherwise
    """
    return user_id not in user_sessions


def mark_user_as_known(user_id: str) -> None:
    """
    Mark user as known (not new anymore).
    
    Args:
        user_id (str): User identifier
    """
    user_sessions[user_id] = True


def store_last_bot_response(user_id: str, response: str) -> None:
    """
    Store the last bot response for a user.
    
    Args:
        user_id (str): User identifier
        response (str): Bot response to store
    """
    if user_id:
        last_bot_responses[user_id] = response


def get_last_bot_response(user_id: str) -> Optional[str]:
    """
    Get the last bot response for a user.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        Optional[str]: Last bot response or None if not found
    """
    return last_bot_responses.get(user_id)


def clear_last_bot_response(user_id: str) -> None:
    """Clear the last bot response for a user."""
    if user_id in last_bot_responses:
        del last_bot_responses[user_id]


def load_faq_data() -> List[Dict[str, str]]:
    """
    Load FAQ data from the JSON file.
    
    Returns:
        List[Dict[str, str]]: List of FAQ items with question and answer
    """
    try:
        # Get the path to the static directory
        static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
        faq_file_path = os.path.join(static_dir, 'faq_data.json')
        
        with open(faq_file_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        return faq_data
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return []


def get_faq_list() -> str:
    """
    Generate a formatted list of all FAQ questions.
    
    Returns:
        str: Formatted FAQ list
    """
    faq_data = load_faq_data()
    if not faq_data:
        return "Maaf, daftar pertanyaan umum tidak tersedia saat ini."
    
    print(f"DEBUG: Loaded {len(faq_data)} FAQ items")
    print(f"DEBUG: First FAQ item: {faq_data[0] if faq_data else 'None'}")
    
    faq_list = "Berikut adalah daftar pertanyaan yang sering diajukan:\n\n"
    
    for i, faq_item in enumerate(faq_data, 1):
        line = f"{i}. {faq_item['question']}\n"
        faq_list += line
        print(f"DEBUG: Added line {i}: {line.strip()}")
    
    faq_list += "\nSilakan balas dengan NOMOR pertanyaan yang Anda inginkan "
    faq_list += "(contoh: 2)."
    
    print(f"DEBUG: Final FAQ list length: {len(faq_list)}")
    debug_chars = repr(faq_list[:300])
    print(f"DEBUG: First 300 chars: {debug_chars}")
    
    return faq_list


def get_faq_answer(question_number: int) -> Optional[str]:
    """
    Get the answer for a specific FAQ question by number.
    
    Args:
        question_number (int): The question number (1-based)
        
    Returns:
        Optional[str]: The answer or None if invalid number
    """
    faq_data = load_faq_data()
    
    if not faq_data or question_number < 1 or question_number > len(faq_data):
        return None
    
    return faq_data[question_number - 1]['answer']


def set_user_faq_context(user_id: str, context: str) -> None:
    """
    Set the FAQ context for a user.
    
    Args:
        user_id (str): User identifier
        context (str): Context value ('awaiting_faq_selection' or None)
    """
    if context:
        user_faq_context[user_id] = context
    elif user_id in user_faq_context:
        del user_faq_context[user_id]


def get_user_faq_context(user_id: str) -> Optional[str]:
    """
    Get the FAQ context for a user.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        Optional[str]: Current FAQ context or None
    """
    return user_faq_context.get(user_id)


def format_bot_response(
    answer: str, 
    is_successful_answer: bool = False, 
    is_faq_response: bool = False
) -> str:
    """
    Minimal post-processing: strip whitespace, remove XML tags, and ensure a friendly closing if missing.
    
    Args:
        answer (str): The response text to format
        is_successful_answer (bool): Whether this is a successful answer from knowledge base
        is_faq_response (bool): Whether this is an FAQ-related response (to avoid adding FAQ trigger)
    """
    # For FAQ responses, preserve the exact formatting but still fix links
    if is_faq_response:
        # Only normalize line breaks but preserve structure
        answer = answer.replace("\r\n", "\n").replace("\r", "\n")
        # Remove any leftover <context>...</context> tags
        answer = re.sub(r"<context>[\s\S]*?</context>", "", answer, flags=re.IGNORECASE)
        # Fix URL formatting even for FAQ responses
        answer = format_links_for_chat(answer)
        return answer
    
    # For regular responses, apply full formatting
    # Remove leading/trailing whitespace and normalize line breaks
    answer = answer.strip().replace("\r\n", "\n").replace("\r", "\n")
    # Remove any leftover <context>...</context> tags
    answer = re.sub(r"<context>[\s\S]*?</context>", "", answer, flags=re.IGNORECASE).strip()
    # Optionally, add a friendly closing if not present
    if not re.search(r"\?\s*$", answer):
        answer += "\n\nApakah ada pertanyaan lain yang bisa saya bantu? 😊"
    
    # Add FAQ trigger footer (except for FAQ-related responses to prevent loops)
    if not is_faq_response:
        faq_trigger = "\n\n----\nKetik \"Menu FAQ\" untuk daftar pertanyaan umum."
        answer += faq_trigger
    
    # Fix URL formatting for chat
    answer = format_links_for_chat(answer)
    
    return answer


@traceable(
    run_type="chain", 
    name="Explain More Request Handler"
)
def handle_explain_more_request(user_id: Optional[str]) -> str:
    """
    Handle requests for more detailed explanations of the last bot response.
    
    Args:
        user_id (Optional[str]): User identifier
        
    Returns:
        str: Detailed explanation or error message
    """
    if not user_id:
        return format_bot_response(
            "Tentu, jelaskan apa yang ingin Anda tanyakan lebih lanjut?",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    last_response = get_last_bot_response(user_id)
    if not last_response:
        return format_bot_response(
            "Tentu, jelaskan apa yang ingin Anda tanyakan lebih lanjut?",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    # Create specialized prompt for elaboration
    elaboration_prompt = f"""You are a helpful assistant. Your previous, concise response to the user was:

---
{last_response}
---

The user has now asked for a more detailed explanation ("Jelaskan lebih jelas"). Your task is to elaborate significantly on your previous answer. Break down complex concepts, provide examples, use analogies, and explain the 'why' behind the information. Do not simply rephrase the original answer. Make it deeper and more comprehensive.

Please provide a much more detailed explanation in Indonesian, maintaining the same helpful and professional tone."""
    
    try:
        # Use the same RAG chain but with the elaboration prompt
        rag_chain, vector_store, embeddings, document_chain = create_rag_chain()
        
        # Get the same context documents that were used for the original response
        # We'll use a general query to get relevant context
        hybrid_docs = hybrid_retrieve("", vector_store, embeddings, top_k=6)
        context_docs = [
            doc for doc in hybrid_docs
            if hasattr(doc, "page_content") and isinstance(doc.page_content, str)
        ]
        
        # Generate detailed response
        detailed_answer = document_chain.invoke({
            "input": elaboration_prompt, 
            "documents": context_docs[:5]
        })
        
        if not detailed_answer or detailed_answer.strip() == "":
            return format_bot_response(
                "Maaf, saya tidak dapat memberikan penjelasan lebih detail saat ini. "
                "Silakan ajukan pertanyaan baru.",
                is_successful_answer=False,
                is_faq_response=False
            )
        
        return format_bot_response(
            detailed_answer, 
            is_successful_answer=True, 
            is_faq_response=False
        )
        
    except Exception as e:
        print(f"Error in handle_explain_more_request: {e}")
        return format_bot_response(
            "Maaf, terjadi kesalahan saat memberikan penjelasan lebih detail. "
            "Silakan coba lagi.",
            is_successful_answer=False,
            is_faq_response=False
        )


@traceable(
    run_type="chain", 
    name="UIN Jakarta TI Chatbot Response"
)
def get_response(query: str, user_id: Optional[str] = None, conversation_has_started: bool = False, is_initial_greeting_sent: bool = False) -> str:
    # Handle empty or None query first
    if not query or not query.strip():
        return format_bot_response(
            "Halo! 👋 \n\nSelamat datang di Customer Service Program Studi "
            "Teknik Informatika UIN Syarif Hidayatullah Jakarta. \n\n"
            "Saya siap membantu Anda dengan informasi seputar kurikulum, "
            "mata kuliah, dosen, dan administrasi akademik. \n\n"
            "Silakan ajukan pertanyaan spesifik tentang informasi yang "
            "Anda butuhkan! 😊",
            is_successful_answer=False,
            is_faq_response=False
        )

        query_lower = query.lower().strip()

    # Handle greetings first
    if query_lower in [
            "hi", "hello", "halo", "hai", "selamat pagi", "selamat siang", "selamat malam"
        ]:
            return format_bot_response(
                "Halo! 👋 \n\nSelamat datang di Customer Service Program Studi "
                "Teknik Informatika UIN Syarif Hidayatullah Jakarta. \n\n"
                "Saya siap membantu Anda dengan informasi seputar kurikulum, "
                "mata kuliah, dosen, dan administrasi akademik. \n\n"
                "Silakan ajukan pertanyaan spesifik tentang informasi yang "
                "Anda butuhkan! 😊",
                is_successful_answer=False,
                is_faq_response=False
            )

    # Handle FAQ commands (moved to top, no longer conditional on user_id)
    if query_lower in ["menu faq", "faq", "pertanyaan umum", "daftar pertanyaan"]:
        faq_list = get_faq_list()
        if user_id:
            set_user_faq_context(user_id, "awaiting_faq_selection")
        return format_bot_response(faq_list, is_successful_answer=False, is_faq_response=True)

    # Handle FAQ number selection (only if user_id is provided)
    if user_id:
        current_faq_context = get_user_faq_context(user_id)
        if current_faq_context == "awaiting_faq_selection":
            try:
                # Try to parse the input as a number
                question_number = int(query.strip())
                faq_answer = get_faq_answer(question_number)
                
                if faq_answer:
                    # Reset FAQ context after providing answer
                    set_user_faq_context(user_id, None)
                    return format_bot_response(faq_answer, is_successful_answer=True, is_faq_response=True)
                else:
                    return format_bot_response(
                        f"Nomor {question_number} tidak valid. Silakan pilih nomor dari daftar di atas.",
                        is_successful_answer=False,
                        is_faq_response=True
                    )
            except ValueError:
                # If not a number, reset context and continue with normal processing
                set_user_faq_context(user_id, None)
                # Continue to normal processing below

        # Check for "Explain More" triggers
        explain_more_triggers = [
            "jelaskan lebih jelas",
            "explain more",
            "tell me more",
            "go into more detail",
            "jelaskan lebih detail",
            "jelaskan lebih lanjut",
            "jelaskan lebih rinci",
            "jelaskan lebih lengkap",
            "saya ingin penjelasan lebih lanjut",
            "can you elaborate",
            "give me more details",
            "be more specific",
            "tell me more about that",
            "in more detail, please",
            "elaborate on that"
        ]
        
        if any(query_lower.startswith(trigger) for trigger in explain_more_triggers):
            return handle_explain_more_request(user_id)

    # Main RAG pipeline logic - only executed if no special commands matched
    try:
        # Main knowledge base answer logic
        rag_chain, vector_store, embeddings, document_chain = create_rag_chain()
        hybrid_docs = hybrid_retrieve(query, vector_store, embeddings, top_k=6)
        context_docs = [
            doc for doc in hybrid_docs
            if hasattr(doc, "page_content") and isinstance(doc.page_content, str)
        ]

        if not context_docs:
            return format_bot_response(
                "Tolong perjelas terkait pertanyaan yang Anda berikan.",
                is_successful_answer=False,
                is_faq_response=False
            )
        answer = document_chain.invoke({"input": query, "documents": context_docs[:5]})
        if not answer or answer.strip() == "":
            return format_bot_response(
                "Tolong perjelas terkait pertanyaan yang Anda berikan.",
                is_successful_answer=False,
                is_faq_response=False
            )
        
        # Store the bot response for "Explain More" feature
        formatted_answer = format_bot_response(answer, is_successful_answer=True, is_faq_response=False)
        store_last_bot_response(user_id, formatted_answer)
        
        return formatted_answer
    except Exception as e:
        print("=== FULL TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        error_msg = (
            f"Maaf, saya mengalami kesalahan dalam memproses "
            f"pertanyaan Anda: {str(e)}"
        )
        return format_bot_response(error_msg, is_successful_answer=False, is_faq_response=False)





def load_kb_files() -> List[Document]:
    """
    Load all knowledge base files from the database and return as Document objects.
    """
    documents: List[Document] = []
    files = KnowledgeBaseFile.query.all()
    for kb_file in files:
        try:
            if kb_file.filetype == "pdf":
                loader = PyMuPDFLoader(kb_file.filepath)
                docs = loader.load()
                for d in docs:
                    d.metadata["file_type"] = "pdf"
                documents.extend(docs)
            elif kb_file.filetype == "txt":
                with open(kb_file.filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": kb_file.filename, "file_type": "txt"},
                        )
                    )
            elif kb_file.filetype == "csv":
                df = pd.read_csv(kb_file.filepath)
                for index, row in df.iterrows():
                    row_text = ""
                    for column, value in row.items():
                        if pd.notna(value):
                            row_text += f"{column}: {value}\n"
                    if row_text.strip():
                        metadata = {
                            "source": kb_file.filename,
                            "row": int(str(index)) + 1,
                            "file_type": "csv",
                        }
                        documents.append(
                            Document(page_content=row_text.strip(), metadata=metadata)
                        )
        except Exception as e:
            print(f"Error loading {kb_file.filename}: {e}")
    return documents


def split_documents_by_type(
    documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 400
) -> List[Document]:
    csv_docs = [doc for doc in documents if doc.metadata.get("file_type") == "csv"]
    pdf_docs = [doc for doc in documents if doc.metadata.get("file_type") == "pdf"]
    txt_docs = [doc for doc in documents if doc.metadata.get("file_type") == "txt"]
    chunks: List[Document] = []
    if csv_docs:
        csv_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks.extend(csv_splitter.split_documents(csv_docs))
    if pdf_docs:
        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks.extend(pdf_splitter.split_documents(pdf_docs))
    if txt_docs:
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks.extend(txt_splitter.split_documents(txt_docs))
    return chunks


def get_changed_files() -> List[KnowledgeBaseFile]:
    """
    Return a list of KnowledgeBaseFile objects whose file hash does not match
    the current file content.
    """
    changed: List[KnowledgeBaseFile] = []
    files = KnowledgeBaseFile.query.all()
    for kb_file in files:
        try:
            with open(kb_file.filepath, "rb") as f:
                filehash = hashlib.sha256(f.read()).hexdigest()
            if filehash != kb_file.filehash:
                changed.append(kb_file)
        except Exception:
            # If file missing or unreadable, treat as changed
            changed.append(kb_file)
    return changed


def run_embedding_background(app: Any, force_all: bool = False) -> None:
    global embedding_progress
    with app.app_context():
        embedding_progress["status"] = "running"
        embedding_progress["progress"] = 0
        embedding_progress["message"] = "Loading documents..."
        if force_all:
            files = KnowledgeBaseFile.query.all()
        else:
            files = get_changed_files()
        total = len(files)
        embedding_progress["total"] = total
        embedding_progress["current"] = 0
        if total == 0:
            embedding_progress["status"] = "done"
            embedding_progress["progress"] = 100
            embedding_progress["message"] = "No files need re-embedding."
            return
        # Load and split only the selected files
        documents: List[Document] = []
        for kb_file in files:
            try:
                if kb_file.filetype == "pdf":
                    loader = PyMuPDFLoader(kb_file.filepath)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["file_type"] = "pdf"
                    documents.extend(docs)
                elif kb_file.filetype == "txt":
                    with open(kb_file.filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": kb_file.filename,
                                    "file_type": "txt",
                                },
                            )
                        )
                elif kb_file.filetype == "csv":
                    df = pd.read_csv(kb_file.filepath)
                    for index, row in df.iterrows():
                        row_text = ""
                        for column, value in row.items():
                            if pd.notna(value):
                                row_text += f"{column}: {value}\n"
                        if row_text.strip():
                            metadata = {
                                "source": kb_file.filename,
                                "row": int(str(index)) + 1,
                                "file_type": "csv",
                            }
                            documents.append(
                                Document(
                                    page_content=row_text.strip(), metadata=metadata
                                )
                            )
            except Exception as e:
                print(f"Error loading {kb_file.filename}: {e}")
            embedding_progress["current"] += 1
            embedding_progress["progress"] = int(
                (embedding_progress["current"]) / total * 100
            )
        embedding_progress["message"] = "Splitting documents..."
        chunks = split_documents_by_type(documents, chunk_size=2000, chunk_overlap=400)
        embedding_progress["message"] = "Creating vector store..."
        create_vector_store(chunks)
        # Update hashes in DB for changed files
        for kb_file in files:
            try:
                with open(kb_file.filepath, "rb") as f:
                    kb_file.filehash = hashlib.sha256(f.read()).hexdigest()
                db.session.commit()
            except Exception:
                pass
        embedding_progress["progress"] = 100
        embedding_progress["status"] = "done"
        embedding_progress["message"] = "Embedding complete!"


def start_embedding(app: Any, force_all: bool = False) -> bool:
    """
    Start the embedding process in a background thread.
    """
    global embedding_progress
    if embedding_progress["status"] == "running":
        return False  # Already running
    embedding_progress["status"] = "starting"
    thread = threading.Thread(target=run_embedding_background, args=(app, force_all))
    thread.start()
    return True


def get_embedding_progress() -> Dict[str, Any]:
    """
    Return the current embedding progress as a dict.
    """
    global embedding_progress
    return embedding_progress


def get_file_status() -> List[Dict[str, Any]]:
    """
    Return a list of dicts with file info and changed status for dashboard display.
    """
    files = KnowledgeBaseFile.query.all()
    changed_ids = {f.id for f in get_changed_files()}
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "filetype": f.filetype,
            "uploaded_at": f.uploaded_at.strftime("%Y-%m-%d %H:%M"),
            "changed": f.id in changed_ids,
        }
        for f in files
    ]
