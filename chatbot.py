import os
import sys
import time
import hashlib
import requests
import logging
import streamlit as st

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain_groq import ChatGroq

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROK_API_KEY")

if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets["GROK_API_KEY"]

MODEL = "llama-3.3-70b-versatile"

BASE_URL = "https://iatnetworks.com"
MAX_PAGES = 40
VECTOR_STORE_PATH = "faiss_index"

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# WEB CRAWLER
# ─────────────────────────────────────────────
visited = set()


def is_valid_url(url):
    base = urlparse(BASE_URL)
    target = urlparse(url)
    return base.netloc == target.netloc


def get_links(url):
    links = set()
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(url, href)

            if is_valid_url(full):
                clean = full.split("#")[0]
                if not any(ext in clean for ext in [".pdf", ".jpg", ".png"]):
                    links.add(clean)

    except Exception as e:
        logging.warning(f"Link error: {url} | {e}")

    return links


def scrape_page(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer", "img"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)

    except Exception as e:
        logging.warning(f"Scrape error: {url} | {e}")
        return ""


def crawl_website():
    queue = [BASE_URL]
    pages = []
    seen_hashes = set()

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)

        if url in visited:
            continue

        logging.info(f"Scraping: {url}")
        visited.add(url)

        content = scrape_page(url)

        if content:
            # ── Fix #5: Deduplicate via MD5 hash ──
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                logging.info(f"Skipping duplicate: {url}")
            else:
                seen_hashes.add(content_hash)
                pages.append((url, content))   # ── Fix #3: store (url, content) pairs

        for link in get_links(url):
            if link not in visited:
                queue.append(link)

        time.sleep(0.5)

    return pages


# ─────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────
class Chatbot:
    def __init__(self, force_rebuild=False):

        # ── Fix #2: Use official ChatGroq ──
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL,
            temperature=0.3
        )

        embedding = HuggingFaceEmbeddings()

        # ── Fix #4: Persist vector store, skip rebuild if already exists ──
        if not force_rebuild and os.path.exists(VECTOR_STORE_PATH):
            print("📂 Loading existing vector store...")
            vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH,
                embedding,
                allow_dangerous_deserialization=True
            )
        else:
            print("🔄 Crawling website...")
            pages = crawl_website()

            print("✂️  Chunking and tagging with source URLs...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            # ── Fix #3: Build Document objects with URL metadata ──
            docs = []
            for url, content in pages:
                chunks = splitter.split_text(content)
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": url}
                    ))

            print(f"📦 Total chunks: {len(docs)}")

            print("🧠 Building embeddings and vector store...")
            vectorstore = FAISS.from_documents(docs, embedding)
            vectorstore.save_local(VECTOR_STORE_PATH)
            print("💾 Vector store saved to disk.")

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ── Fix #1: Use ConversationSummaryBufferMemory to cap prompt size ──
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            memory_key="chat_history",
            return_messages=True
        )

    def ask(self, question):
        # Load summarized history
        history = self.memory.load_memory_variables({})["chat_history"]
        history_text = "\n".join([str(h) for h in history])

        # ── Fix #6: Use .invoke() instead of deprecated get_relevant_documents() ──
        retrieved_docs = self.retriever.invoke(question)

        # ── Fix #3: Include source URLs in context ──
        context = "\n\n".join([
            f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        prompt = f"""
You are a chatbot for IAT Networks.

STRICT RULES:
- ONLY answer questions related to IAT Networks.
- Use ONLY the provided context.
- DO NOT answer general knowledge questions.
- DO NOT mention training data, AI model, or limitations.
- If user greets (hi, hello, thanks, etc.), respond politely.

If the question is about IAT Networks BUT the context does not contain the answer:
Respond with:
"For the most accurate information, please contact us at hr@iatnetworks.com."

If the question is not related to IAT Networks, respond EXACTLY with:
"I'm here to help with questions about IAT Networks. Please ask something relevant to our organization."

━━━━━━━━━━━━━━━━━━━━━━━
Conversation History:
{history_text}

Context:
{context}

Question:
{question}
━━━━━━━━━━━━━━━━━━━━━━━

Answer:
"""

        response = self.llm.invoke(prompt)
        answer = response.content.strip()

        # Save turn to summarized memory
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )

        return answer


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("⚠️  Set GROQ_API_KEY as an environment variable.")
        sys.exit(1)

    # Pass --rebuild flag to force re-crawl and re-embed
    force_rebuild = "--rebuild" in sys.argv
    bot = Chatbot(force_rebuild=force_rebuild)

    print("\n✅ Chatbot ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("🧑 You: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        start = time.time()
        answer = bot.ask(question)
        elapsed = round((time.time() - start) * 1000)

        print(f"\n🤖 Bot: {answer}")
        print(f"⏱️  {elapsed} ms\n")