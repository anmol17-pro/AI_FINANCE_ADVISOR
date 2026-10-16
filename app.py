import os
import time
import requests
import gradio as gr
import assemblyai as aai
from murf import Murf

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Voice APIs are optional for text-only use.
# Voice input requires ASSEMBLYAI_API_KEY.
# Voice output requires MURF_API_KEY.
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

MURF_VOICE_ID = os.getenv("MURF_VOICE_ID", "Natalie")
MURF_LOCALE = os.getenv("MURF_LOCALE", "en-US")
AUDIO_DIR = "audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)

murf_client = Murf(api_key=MURF_API_KEY) if MURF_API_KEY else None


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEBI_PDF = (
    "https://investor.sebi.gov.in/pdf/"
    "downloadable-documents/"
    "Financial%20Education%20Booklet%20-%20English.pdf"
)

GST_PDF = (
    "https://a2ztaxcorp.net/wp-content/uploads/2025/09/"
    "CBIC-GST-Ready-Reckoner-indicating-updated-"
    "Central-Goods-and-Services-Tax-CGST-rates-on-goods.pdf"
)

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


# ============================================================
# 3. DOWNLOAD PDF
# ============================================================

def download_pdf(url, filename):

    os.makedirs(DATA_DIR, exist_ok=True)

    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        print(f"Already exists: {path}")
        return path

    print(f"Downloading {filename}...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(path, "wb") as file:
        file.write(response.content)

    print(f"Downloaded: {path}")

    return path


# ============================================================
# 4. LOAD FINANCIAL DOCUMENTS
# ============================================================

def load_documents():

    sebi_path = download_pdf(
        SEBI_PDF,
        "sebi_financial_education.pdf"
    )

    gst_path = download_pdf(
        GST_PDF,
        "gst_ready_reckoner.pdf"
    )

    all_documents = []

    print("Loading SEBI document...")

    sebi_loader = PyPDFLoader(sebi_path)
    all_documents.extend(sebi_loader.load())

    print("Loading GST document...")

    gst_loader = PyPDFLoader(gst_path)
    all_documents.extend(gst_loader.load())

    print(
        f"Loaded {len(all_documents)} pages."
    )

    return all_documents


# ============================================================
# 5. CREATE VECTOR DATABASE
# ============================================================

def create_vector_store():

    documents = load_documents()

    print("Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Creating Chroma vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="finance_documents"
    )

    print("Vector database ready.")

    return vector_store


# ============================================================
# 6. INITIALIZE VECTOR DATABASE
# ============================================================

vector_store = create_vector_store()


# ============================================================
# 7. DOCUMENT SEARCH TOOL
# ============================================================

@tool
def search_finance_docs(question: str) -> str:
    """
    Search Indian financial documents for information about
    GST, taxation, investment guidance, insurance,
    government schemes, financial education and related topics.
    """

    results = vector_store.similarity_search(
        question,
        k=4
    )

    if not results:
        return "No relevant information was found."

    context_parts = []

    for doc in results:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        )

        page_number = page + 1

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page_number}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# 8. LIVE MARKET PRICE TOOL
# ============================================================

@tool
def get_market_price(asset: str) -> str:
    """
    Get the current market price of gold or silver
    in Indian Rupees per gram.

    Supported assets:
    gold
    silver
    """

    symbols = {
        "gold": "XAU",
        "silver": "XAG"
    }

    asset = asset.lower().strip()

    symbol = symbols.get(asset)

    if not symbol:
        return (
            "Unsupported asset. "
            "Use gold or silver."
        )

    url = (
        f"https://api.gold-api.com/"
        f"price/{symbol}/INR"
    )

    try:

        response = requests.get(
            url,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        price_per_ounce = data["price"]

        price_per_gram = (
            price_per_ounce / 31.1035
        )

        updated = data.get(
            "updatedAtReadable",
            "unknown"
        )

        return (
            f"{asset.title()} current price: "
            f"₹{price_per_gram:,.2f} per gram.\n"
            f"Source: Gold-API.\n"
            f"Last update: {updated}"
        )

    except Exception as e:

        return (
            f"Unable to retrieve {asset} price. "
            f"Error: {str(e)}"
        )


# ============================================================
# 9. Gemini MODEL
# ============================================================

model = init_chat_model(
    model="gemini-3.6-flash",
    model_provider="google_genai",
    api_key=GEMINI_API_KEY,
    temperature=0
)

# ============================================================
# 10. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a Personal Finance AI Advisor for Indian citizens.

Your job is to provide useful, careful, simple financial education.

AVAILABLE TOOLS:

1. search_finance_docs
Use this for:
- GST
- taxation
- investment education
- insurance
- government schemes
- financial education
- information contained in the provided documents

2. get_market_price
Use this for:
- current gold price
- current silver price

IMPORTANT RULES:
- Always use get_market_price for current gold/silver prices.
- Never invent a live market price.
- Use search_finance_docs for document-based financial information.
- If a question requires both current market data and document information, use both tools.
- Clearly explain where information came from.
- Use Indian Rupees (₹) for monetary values.
- Do not pretend to know information that the tools do not provide.
- Give simple explanations suitable for normal Indian users.
- For financial decisions, clearly distinguish factual information from general educational guidance.

OUTPUT FORMAT:
Always make the final answer easy to scan using Markdown.

Use this structure whenever it fits the question:

## 💡 Quick Answer
Give the direct answer in 1-3 sentences.

## 📌 Key Points
- Give the most important points.
- Use short bullets.
- Include ₹ values when relevant.

## 📊 Details
Explain calculations, rules, comparisons, or reasoning when needed.

## ⚠️ Important
Mention assumptions, limitations, risks, or things the user should verify.

## 📚 Source
State whether the answer came from:
- SEBI/Government/financial documents
- Live market data
- General educational guidance

Do not include empty sections. Do not use unnecessary long paragraphs.
"""



# ============================================================
# 11. CREATE AI AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=[
        search_finance_docs,
        get_market_price
    ],
    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# ============================================================
# 12. VOICE + TEXT FINANCE ADVISOR
# ============================================================

def transcribe_audio(audio_path):
    """Convert microphone/uploaded audio to text using AssemblyAI."""
    if not audio_path:
        return ""

    if not ASSEMBLYAI_API_KEY:
        raise ValueError(
            "ASSEMBLYAI_API_KEY is missing. Add it to your .env file "
            "to use voice input."
        )

    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(transcript.error)

        return (transcript.text or "").strip()

    except Exception as e:
        raise RuntimeError(f"AssemblyAI transcription failed: {e}")


def generate_murf_audio(text):
    """Convert the final AI answer to speech using Murf."""
    if not text:
        return None

    if not MURF_API_KEY or not murf_client:
        return None

    try:
        # Keep the spoken version shorter and cleaner.
        speech_text = text

        # Remove common Markdown markers so the voice sounds natural.
        for marker in [
            "## 💡 Quick Answer",
            "## 📌 Key Points",
            "## 📊 Details",
            "## ⚠️ Important",
            "## 📚 Source",
            "### ",
            "**",
            "*",
        ]:
            speech_text = speech_text.replace(marker, "")

        speech_text = speech_text.strip()

        # Murf's current non-streaming API returns an audio URL.
        result = murf_client.text_to_speech.generate(
            text=speech_text,
            voice_id=MURF_VOICE_ID,
            locale=MURF_LOCALE,
            format="MP3",
        )

        audio_url = result.audio_file

        if not audio_url:
            raise RuntimeError("Murf did not return an audio file URL.")

        filename = os.path.join(
            AUDIO_DIR,
            f"finance_answer_{int(time.time() * 1000)}.mp3"
        )

        audio_response = requests.get(audio_url, timeout=60)
        audio_response.raise_for_status()

        with open(filename, "wb") as audio_file:
            audio_file.write(audio_response.content)

        return filename

    except Exception as e:
        raise RuntimeError(f"Murf text-to-speech failed: {e}")


def finance_advisor(question, audio_path):
    """
    Main pipeline:

    Microphone/upload audio
          ↓
      AssemblyAI
          ↓
       Text question
          ↓
     LangChain Agent
          ↓
   Finance tools / RAG
          ↓
    Structured answer
          ↓
        Murf TTS
          ↓
       Audio answer
    """

    try:
        # 1. Voice input takes priority if audio was provided.
        transcript = ""

        if audio_path:
            transcript = transcribe_audio(audio_path)
            question = transcript

        if not question or not question.strip():
            return (
                "Please type a question or record your question using "
                "the microphone.",
                "",
                None,
            )

        question = question.strip()

        # 2. Ask the finance agent.
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        answer = response["messages"][-1].content

        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in answer
            )

        answer = str(answer).strip()

        # 3. Generate voice response with Murf.
        audio_output = generate_murf_audio(answer)

        return answer, transcript, audio_output

    except Exception as e:
        error_message = (
            "## ❌ Error\n\n"
            f"{str(e)}\n\n"
            "Please check your API keys and try again."
        )
        return error_message, "", None


# ============================================================
# 13. GRADIO UI
# ============================================================

with gr.Blocks(title="Personal Finance AI Advisor") as demo:

    gr.Markdown(
        """
        # 💰 Personal Finance AI Advisor

        ### 🧠 AI + RAG + Live Market Data + Voice

        Ask your finance question by **typing or speaking**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                lines=5,
                label="⌨️ Type your question",
                placeholder=(
                    "Example: What is the current gold price in India?"
                ),
            )

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Speak your question",
            )

            ask_button = gr.Button(
                "🚀 Ask Finance Advisor",
                variant="primary",
            )

            clear_button = gr.ClearButton(
                components=[
                    question_input,
                    audio_input,
                ],
                value="🧹 Clear",
            )

        with gr.Column(scale=1):
            transcript_output = gr.Textbox(
                lines=5,
                label="📝 AssemblyAI Transcript",
                placeholder="Voice transcript will appear here...",
                interactive=False,
            )

            answer_output = gr.Markdown(
                value="Your structured financial answer will appear here."
            )

            audio_output = gr.Audio(
                label="🔊 Murf AI Voice Answer",
                type="filepath",
                autoplay=False,
            )

    gr.Markdown(
        """
        ### 💡 Example Questions

        - What is the current gold price?
        - What is the current silver price?
        - What GST information is available for laptops?
        - What government investment schemes are discussed?
        - What types of insurance should a family consider?

        ### 🔄 How it works

        **Voice → AssemblyAI Speech-to-Text → Finance AI Agent → RAG/Live Tools → Structured Answer → Murf Text-to-Speech**
        """
    )

    ask_button.click(
        fn=finance_advisor,
        inputs=[question_input, audio_input],
        outputs=[answer_output, transcript_output, audio_output],
    )

# 14. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PERSONAL FINANCE AI ADVISOR")
    print("=" * 60)

    print("Starting Gradio application...")

    demo.launch()