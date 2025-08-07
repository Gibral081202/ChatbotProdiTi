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

# Store last bot response with original query and context for better "Explain More"
last_bot_context: Dict[str, Dict[str, Any]] = {}

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


def format_for_whatsapp(text: str) -> str:
    """
    Format text specifically for WhatsApp to ensure links are clickable.
    
    This function converts Markdown links to plain text URLs and cleans up formatting
    that might interfere with WhatsApp's link detection.
    
    Args:
        text (str): The response text to format for WhatsApp
        
    Returns:
        str: Text formatted for WhatsApp with clean, clickable URLs
    """
    if not text:
        return text
    
    # Pattern to match Markdown links: [text](url) or [url](url)
    # This will match patterns like:
    # [Link Text](https://example.com)
    # [https://example.com](https://example.com)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    def replace_markdown_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        
        # Always return just the URL for WhatsApp
        return link_url
    
    # Replace all Markdown links with just the URL
    formatted_text = re.sub(markdown_link_pattern, replace_markdown_link, text)
    
    # Remove list markers (*, -, etc.) and extra whitespace from each line
    lines = formatted_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove leading list markers and whitespace
        cleaned_line = re.sub(r'^[\s\-\*\+]+', '', line.strip())
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
    
    # Join lines back together
    formatted_text = '\n'.join(cleaned_lines)
    
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
        """Anda adalah asisten AI layanan profesional, ramah, dan ahli untuk Program Studi Teknik Informatika UIN Syarif Hidayatullah Jakarta. Tugas Anda adalah menjawab pertanyaan berdasarkan informasi yang diberikan dalam <context> berikut.

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


def store_last_bot_context(user_id: str, query: str, response: str, context_docs: List[Document]) -> None:
    """
    Store the last bot response with its original query and context documents.
    This OVERWRITES any previous context to ensure we always have the most recent.
    
    Args:
        user_id (str): User identifier
        query (str): Original user query
        response (str): Bot response
        context_docs (List[Document]): Context documents used for the response
    """
    if user_id:
        # Always overwrite the previous context to ensure we have the most recent
        last_bot_context[user_id] = {
            "query": query,
            "response": response,
            "context_docs": context_docs,
            "timestamp": __import__('time').time()
        }
        print(f"[CONTEXT] Stored new context for user {user_id}: '{query[:50]}...'")


def get_last_bot_context(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the last bot context for a user.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        Optional[Dict]: Last bot context or None if not found
    """
    return last_bot_context.get(user_id)


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
    if user_id in last_bot_context:
        del last_bot_context[user_id]


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
    try:
        # Load FAQ data
        faq_data = load_faq_data()
        print(f"[FAQ] Attempting to get answer for number {question_number}")
        print(f"[FAQ] Total FAQ items loaded: {len(faq_data)}")
        
        # Validate FAQ data
        if not faq_data:
            print(f"[FAQ] Error: FAQ Data is empty or not loaded")
            return None
            
        # Validate question number
        if question_number < 1 or question_number > len(faq_data):
            print(f"[FAQ] Error: Invalid FAQ number {question_number}. Valid range: 1-{len(faq_data)}")
            return None
        
        # Get the FAQ item
        try:
            faq_item = faq_data[question_number - 1]
            print(f"[FAQ] Found FAQ item: {faq_item.get('question', 'NO QUESTION')} ({question_number})")
        except IndexError:
            print(f"[FAQ] Error: Could not access FAQ item at index {question_number - 1}")
            return None
        
        # Get and validate answer
        answer = faq_item.get('answer')
        if not answer or not isinstance(answer, str):
            print(f"[FAQ] Error: Invalid or missing answer for FAQ number {question_number}")
            return None
            
        # Success!
        print(f"[FAQ] Successfully retrieved answer for FAQ number {question_number}")
        print(f"[FAQ] Answer preview: {answer[:100]}...")
        return answer
        
    except Exception as e:
        print(f"[FAQ] Critical error getting FAQ answer for number {question_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


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
    
    # Add combined footer with both "Explain More" and "Menu FAQ" options
    # (except for FAQ-related responses to prevent loops)
    if not is_faq_response:
        footer = "\n\n----\nKetik:\n• \"Jelaskan Lebih Jelas\" untuk rincian.\n• \"Menu FAQ\" untuk daftar pertanyaan umum."
        answer += footer
    
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
            "Maaf, saya tidak memiliki respons sebelumnya untuk dijelaskan lebih detail. "
            "Silakan ajukan pertanyaan spesifik yang ingin Anda ketahui.",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    # Get the last bot context (includes original query, response, and documents)
    print(f"[EXPLAIN] Getting context for user_id: {user_id}")
    last_context = get_last_bot_context(user_id)
    if not last_context:
        print(f"[EXPLAIN] No context found for user_id: {user_id}")
        return format_bot_response(
            "Maaf, saya tidak memiliki respons sebelumnya untuk dijelaskan lebih detail. "
            "Silakan ajukan pertanyaan baru yang ingin Anda ketahui.",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    # Validate that we have the required context data
    if not last_context.get("query") or not last_context.get("response"):
        return format_bot_response(
            "Maaf, konteks respons sebelumnya tidak lengkap. "
            "Silakan ajukan pertanyaan baru yang ingin Anda ketahui.",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    # Check if the context is still recent (within 10 minutes)
    import time
    current_time = time.time()
    context_age = current_time - last_context.get("timestamp", 0)
    if context_age > 600:  # 10 minutes
        return format_bot_response(
            "Maaf, respons sebelumnya sudah terlalu lama. "
            "Silakan ajukan pertanyaan baru yang ingin Anda ketahui.",
            is_successful_answer=False,
            is_faq_response=False
        )
    
    original_query = last_context.get("query", "")
    last_response = last_context.get("response", "")
    context_docs = last_context.get("context_docs", [])
    
    print(f"[EXPLAIN] Using context for query: '{original_query}'")
    print(f"[EXPLAIN] Context age: {context_age:.1f} seconds")
    print(f"[EXPLAIN] Response preview: {last_response[:100]}...")
    
    # Create specialized prompt for elaboration
    elaboration_prompt = f"""Anda adalah asisten AI layanan profesional untuk Program Studi Teknik Informatika UIN Syarif Hidayatullah Jakarta.

PERTANYAAN ASLI PENGGUNA: "{original_query}"

RESPONS ANDA SEBELUMNYA:
---
{last_response}
---

PENGGUNA SEKARANG MEMINTA PENJELASAN YANG LEBIH DETAIL ("Jelaskan Lebih Jelas") tentang respons Anda di atas.

TUGAS ANDA:
1. Berikan penjelasan yang LEBIH DETAIL dan MENDALAM tentang respons sebelumnya
2. Pecah konsep-konsep kompleks menjadi bagian-bagian yang mudah dipahami
3. Berikan contoh konkret jika memungkinkan
4. Jelaskan "mengapa" di balik informasi yang diberikan
5. Gunakan HANYA informasi dari konteks dokumen yang sama
6. Tetap fokus pada topik respons sebelumnya
7. Gunakan format Markdown yang rapi
8. Jawab dalam Bahasa Indonesia yang profesional

PENTING: 
- Jangan menambahkan informasi baru di luar konteks respons sebelumnya
- Fokus hanya pada penjelasan lebih detail dari apa yang sudah dijelaskan
- Pastikan penjelasan Anda terkait dengan pertanyaan: "{original_query}"
- Jika ada informasi yang tidak relevan dengan pertanyaan asli, abaikan"""
    
    try:
        # Use the same context documents from the original response
        if not context_docs:
            return format_bot_response(
                "Maaf, saya tidak dapat memberikan penjelasan lebih detail karena "
                "tidak ada konteks dokumen dari respons sebelumnya. "
                "Silakan ajukan pertanyaan baru.",
                is_successful_answer=False,
                is_faq_response=False
            )
        
        # Load the LLM and document chain
        rag_chain, vector_store, embeddings, document_chain = create_rag_chain()
        
        # Use the SAME context documents from the original response
        valid_docs = [
            doc for doc in context_docs
            if hasattr(doc, "page_content") and isinstance(doc.page_content, str)
        ]
        
        if not valid_docs:
            return format_bot_response(
                "Maaf, saya tidak dapat memberikan penjelasan lebih detail saat ini. "
                "Silakan ajukan pertanyaan baru.",
                is_successful_answer=False,
                is_faq_response=False
            )
        
        # Generate detailed response using the same context
        detailed_answer = document_chain.invoke({
            "input": elaboration_prompt, 
            "documents": valid_docs[:5]
        })
        
        if not detailed_answer or detailed_answer.strip() == "":
            return format_bot_response(
                "Maaf, saya tidak dapat memberikan penjelasan lebih detail saat ini. "
                "Silakan ajukan pertanyaan baru.",
                is_successful_answer=False,
                is_faq_response=False
            )
        
        # Store this detailed response as the new last response
        formatted_answer = format_bot_response(
            detailed_answer, 
            is_successful_answer=True, 
            is_faq_response=False
        )
        
        # Update the stored context with the new detailed response
        # This ensures that if user asks "Jelaskan Lebih Jelas" again, they get even more detail
        store_last_bot_context(user_id, original_query, formatted_answer, context_docs)
        
        print(f"[EXPLAIN] Successfully generated detailed explanation for: '{original_query}'")
        return formatted_answer
        
    except Exception as e:
        print(f"Error in handle_explain_more_request: {e}")
        import traceback
        traceback.print_exc()
        return format_bot_response(
            "Maaf, terjadi kesalahan saat memberikan penjelasan lebih detail. "
            "Silakan coba lagi atau ajukan pertanyaan baru.",
            is_successful_answer=False,
            is_faq_response=False
        )


@traceable(
    run_type="chain", 
    name="UIN Jakarta TI Chatbot Response"
)
def get_response(query: str, user_id: Optional[str] = None, conversation_has_started: bool = False, is_initial_greeting_sent: bool = False) -> str:
    # 1. Handle Empty/None Queries First
    if not query or not query.strip():
        # Clear any previous context for fresh start
        if user_id:
            clear_last_bot_response(user_id)
        return format_bot_response(
            "Halo! 👋 \n\nLayanan Program Studi "
            "Teknik Informatika UIN Syarif Hidayatullah Jakarta. \n\n"
            "Saya siap membantu Anda dengan informasi seputar kurikulum, "
            "mata kuliah, dosen, dan administrasi akademik. \n\n"
            "Silakan ajukan pertanyaan spesifik tentang informasi yang "
            "Anda butuhkan! 😊",
            is_successful_answer=False,
            is_faq_response=False
        )

    # 2. Define query_lower Immediately
    query_lower = query.lower().strip()

    # 3. Handle All Special Commands
    # Handle greetings
    if query_lower in [
        "hi", "hello", "halo", "hai", "selamat pagi", "selamat siang", "selamat malam"
    ]:
        # Clear any previous context for fresh start
        if user_id:
            clear_last_bot_response(user_id)
        return format_bot_response(
            "Halo! 👋 \n\nLayanan Program Studi "
            "Teknik Informatika UIN Syarif Hidayatullah Jakarta. \n\n"
            "Saya siap membantu Anda dengan informasi seputar kurikulum, "
            "mata kuliah, dosen, dan administrasi akademik. \n\n"
            "Silakan ajukan pertanyaan spesifik tentang informasi yang "
            "Anda butuhkan! 😊",
            is_successful_answer=False,
            is_faq_response=False
        )

    # Handle FAQ commands (no longer conditional on user_id)
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
                print(f"[FAQ] Processing user input: {query}")
                # Clean and normalize input
                query_clean = query.strip().lower()
                query_clean = query_clean.replace("no", "nomor").replace("no.", "nomor")
                query_clean = query_clean.replace(".", "").replace(",", "")
                
                # Handle common number formats
                number_map = {
                    "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
                    "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10",
                    "sebelas": "11", "duabelas": "12", "tigabelas": "13", "empatbelas": "14",
                    "limabelas": "15", "enambelas": "16", "tujuhbelas": "17", "delapanbelas": "18",
                    "sembilanbelas": "19", "duapuluh": "20", "duapuluhsatu": "21", "duapuluhdua": "22",
                    # Add variations
                    "tujuhbelas": "17", "delapanbelas": "18", "tigabelas": "13",
                    "nomor 7": "7", "nomor 13": "13", "nomor 18": "18", "nomor 22": "22"
                }
                
                # First try exact matches with number_map
                if query_clean in number_map:
                    question_number = int(number_map[query_clean])
                    print(f"[FAQ] Matched exact number word: {query_clean} -> {question_number}")
                else:
                    # Try to find number words in the query
                    for word, digit in number_map.items():
                        if word in query_clean:
                            question_number = int(digit)
                            print(f"[FAQ] Found number word in query: {word} -> {question_number}")
                            break
                    else:
                        # If no word numbers found, try to extract digits
                        import re
                        numbers = re.findall(r'\d+', query_clean)
                        if numbers:
                            question_number = int(numbers[0])
                            print(f"[FAQ] Extracted number from query: {question_number}")
                        else:
                            # Last resort: try to convert the whole string
                            try:
                                question_number = int(query_clean)
                                print(f"[FAQ] Converted whole query to number: {question_number}")
                            except ValueError:
                                print(f"[FAQ] Could not extract number from: {query_clean}")
                                return format_bot_response(
                                    "Maaf, saya tidak dapat mengenali nomor pertanyaan yang Anda maksud. "
                                    "Silakan ketik nomor pertanyaan yang ingin Anda tanyakan (contoh: 7).",
                                    is_successful_answer=False,
                                    is_faq_response=True
                                )
                
                print(f"[FAQ] Final question number: {question_number}")
                faq_answer = get_faq_answer(question_number)
                
                if faq_answer:
                    # Reset FAQ context after providing answer
                    set_user_faq_context(user_id, None)
                    formatted_answer = format_bot_response(faq_answer, is_successful_answer=True, is_faq_response=True)
                    print(f"[FAQ] Sending FAQ answer for number {question_number}")
                    
                    # Clear any previous "Explain More" context since FAQ answers are complete
                    clear_last_bot_response(user_id)
                    
                    return formatted_answer
                else:
                    error_msg = f"Nomor {question_number} tidak valid. Silakan pilih nomor dari daftar di atas."
                    print(f"Invalid FAQ number: {question_number}")
                    return format_bot_response(error_msg, is_successful_answer=False, is_faq_response=True)
            except ValueError as e:
                print(f"Error parsing FAQ number from input '{query}': {str(e)}")
                # If not a number, reset context and continue with normal processing
                set_user_faq_context(user_id, None)
                # Continue to normal processing below
            except Exception as e:
                print(f"Unexpected error handling FAQ request: {str(e)}")
                set_user_faq_context(user_id, None)
                return format_bot_response(
                    "Maaf, terjadi kesalahan dalam memproses permintaan FAQ Anda. Silakan coba lagi.",
                    is_successful_answer=False,
                    is_faq_response=True
                )

    # Check for "Explain More" triggers (case-insensitive)
    explain_more_triggers = [
        "jelaskan",
        "jelaskan lebih jelas",
        "jelaskan lebih detail",
        "jelaskan lebih lanjut",
        "jelaskan lebih rinci",
        "jelaskan lebih lengkap",
        "saya ingin penjelasan lebih lanjut",
        "explain more",
        "tell me more",
        "go into more detail",
        "can you elaborate",
        "give me more details",
        "be more specific",
        "tell me more about that",
        "in more detail, please",
        "elaborate on that"
    ]
    
    if any(query_lower.startswith(trigger) for trigger in explain_more_triggers):
        return handle_explain_more_request(user_id)

    # 4. Isolate the RAG Pipeline
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
        
        # Store the complete context for better "Explain More" functionality
        # This OVERWRITES any previous context to ensure we always have the most recent
        store_last_bot_context(user_id, query, formatted_answer, context_docs)
        
        print(f"[RESPONSE] Stored context for user_id: {user_id}, query: '{query[:50]}...'")
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
