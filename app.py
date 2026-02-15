import streamlit as st
import tempfile
import os
from pathlib import Path
from indexing.document_loader import load_document
from indexing.vector_store import add_document
from pipeline.orchestrator import answer_query
from utils.helpers import format_sources

st.set_page_config(page_title="Groq RAG with Thresholds", layout="wide")
st.markdown("""
<style>
    /* User avatar */
    [data-testid="chatAvatarIcon-user"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    /* Assistant avatar */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #424242 !important;
        color: white !important;
    }
    /* Optional: change text bubble colors */
    [data-testid="chat-message-user"] .stChatMessageContent {
        background-color: #E3F2FD !important;
    }
    [data-testid="chat-message-assistant"] .stChatMessageContent {
        background-color: #F5F5F5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
#  Session state initialisation
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

# ----------------------------
#  Sidebar – File Upload
# ----------------------------
st.sidebar.title("📁 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files and st.sidebar.button("Index Documents"):
    with st.sidebar.status("Indexing documents...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            # Save to temporary file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Extract text
            text = load_document(tmp_path)
            os.unlink(tmp_path)  # clean up
            
            # Add to vector store (sentence‑level chunking)
            add_document(
                doc_id=uploaded_file.name,
                text=text,
                metadata={"source": uploaded_file.name}
            )
        status.update(label="Indexing complete!", state="complete")
        st.session_state.vectorstore_ready = True

st.sidebar.divider()
st.sidebar.write("**Current thresholds**")
st.sidebar.info(
    f"UT: {os.getenv('UPPER_THRESHOLD', 8.0)}  \n"
    f"LT: {os.getenv('LOWER_THRESHOLD', 3.0)}  \n"
    f"Strip: {os.getenv('STRIP_THRESHOLD', 5.0)}"
)

# ----------------------------
#  Main Chat Interface
# ----------------------------
st.title("💬 Corrective RAG (CRAG)")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                st.text(msg["sources"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Run pipeline
        response_stream, sources, trace = answer_query(prompt)
        
        # Stream output
        for chunk in response_stream:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
        # Show sources
        if sources:
            with st.expander("📄 Sources"):
                st.text(format_sources(sources))
        else:
            with st.expander("📄 Sources"):
                st.write("No sources were used (fallback).")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": format_sources(sources) if sources else "No sources."
        })
        # After streaming the answer, show trace
        if trace:
            with st.expander("🔍 Pipeline Trace (Internal Details)"):
                st.markdown("#### 📄 Retrieved Chunks")
                for c in trace.get("retrieved_chunks", []):
                    st.markdown(f"- **{c['source']}** (ID: {c['id']})")
                    st.markdown(f"  _Preview:_ {c['text_preview']}...")

                st.markdown("#### Chunk Scores")
                for s in trace.get("chunk_scores", []):
                    st.markdown(f"- {s['source']}: **{s['score']}**")

                st.markdown(f"#### Classification: **{trace['classification']}**")
                if trace["classification"] == "correct":
                    st.success(" Using local documents only")
                elif trace["classification"] == "ambiguous":
                    st.warning("Ambiguous – refined local chunks + web search")
                else:
                    st.error("Incorrect – web search only")

                if trace.get("refined_local"):
                    st.markdown("#### Refined Local Chunks")
                    for r in trace["refined_local"]:
                        st.markdown(f"- {r['chunk_id']}: kept **{r['strips_kept']}** strips")
                        st.markdown(f"  _Preview:_ {r['preview']}...")

                if trace.get("web_search_used"):
                    st.markdown("#### 🌐 Web Search")
                    st.markdown(f"Rewritten query: `{trace.get('rewritten_query', 'N/A')}`")
                    for w in trace.get("web_results", []):
                        st.markdown(f"- [{w['title']}]({w['url']})")
                        st.markdown(f"  _Preview:_ {w['preview']}...")

                st.markdown("#### 📝 Final Context Preview")
                st.text(trace.get("final_context", "No context generated."))

