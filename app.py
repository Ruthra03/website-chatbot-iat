import streamlit as st
from chatbot import Chatbot

st.set_page_config(
    page_title="IAT Networks",
    layout="centered",
)

# Minimal styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }

    .chat-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #202123;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        margin: 0;
    }

    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        background: #f7f7f8;
        color: #353740;
        font-size: 0.85rem;
    }

    div[data-testid="stHorizontalBlock"] button:hover {
        background: #ececf1;
        border-color: #d9d9e3;
    }
</style>
""", unsafe_allow_html=True)

# Default suggestions
DEFAULT_QUESTIONS = [
    "What services does IAT Networks offer?",
    "How can I contact IAT Networks?",
    "What industries does IAT Networks serve?",
]

# Initialize session state
if "chatbot" not in st.session_state:
    with st.spinner("Loading..."):
        st.session_state.chatbot = Chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# Title
st.markdown('<p class="chat-title">IAT Networks</p>', unsafe_allow_html=True)

# Suggested questions (only before chat starts)
if not st.session_state.messages:
    cols = st.columns(3)
    for i, question in enumerate(DEFAULT_QUESTIONS):
        if cols[i].button(question, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending_input = question
            st.rerun()

# Display chat history (NO avatars)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=None):
        st.markdown(msg["content"])

# Handle query
def handle_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar=None):
        st.markdown(query)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("Loading..."):
            answer = st.session_state.chatbot.ask(query)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Handle suggested click
if st.session_state.pending_input:
    query = st.session_state.pending_input
    st.session_state.pending_input = None
    handle_query(query)

# User input box
user_input = st.chat_input("Type your message...")
if user_input:
    handle_query(user_input)