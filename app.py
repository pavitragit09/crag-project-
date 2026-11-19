import streamlit as st
import tempfile
import os
import re
from pathlib import Path
from indexing.document_loader import load_document
from indexing.vector_store import add_document
from pipeline.orchestrator import answer_query
from utils.helpers import format_sources
from core.config import UPPER_THRESHOLD, LOWER_THRESHOLD, STRIP_THRESHOLD


st.set_page_config(page_title="Corrective RAG (CRAG)", layout="wide")
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

def split_response(full_response: str):
    """Separate thinking (inside <think> tags) from the final answer."""
    think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    else:
        thinking = None
        answer = full_response.strip()
    return thinking, answer

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
        try:
            indexed_chunks = 0
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    text = load_document(tmp_path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                indexed_chunks += add_document(
                    doc_id=uploaded_file.name,
                    text=text,
                    metadata={"source": uploaded_file.name},
                )

            status.update(label=f"Indexed {indexed_chunks} chunks.", state="complete")
            st.session_state.vectorstore_ready = indexed_chunks > 0
        except Exception as exc:
            status.update(label="Document indexing failed.", state="error")
            st.sidebar.error(str(exc))

st.sidebar.divider()
st.sidebar.write("**Current thresholds**")
st.sidebar.info(
    f"UT : {UPPER_THRESHOLD:0.7}  \n"
    f"LT : {LOWER_THRESHOLD:0.4}  \n"
    f"Strip (sentence score 0-10): {STRIP_THRESHOLD:g}"
)

# ----------------------------
#  Main App Views
# ----------------------------
active_view = st.radio(
    "App section",
    ["💬 Ask Documents", "📊 Benchmark Metrics"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_view",
)

prompt = None
if active_view == "💬 Ask Documents":
    st.title("💬 Corrective RAG (CRAG)")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    st.text(msg["sources"])

    # Keep chat_input at the page level: Streamlit does not allow it inside tabs.
    prompt = st.chat_input("Ask a question about your documents...")
else:
    st.title("Baseline RAG vs CRAG")
    st.caption("Reported results on the project's 12-question benchmark; these are static report figures, not live measurements.")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Average correctness", "0.97 baseline", help="Baseline RAG: 0.97; CRAG: 0.86")
    metric_cols[1].metric("Average faithfulness", "0.71 baseline", help="Baseline RAG: 0.71; CRAG: 0.75")
    metric_cols[2].metric("Average latency", "4.97 s baseline", help="Baseline RAG: 4.97 s; CRAG: 12.19 s")
    st.table({
        "Metric": ["Average correctness", "Average faithfulness", "Average latency"],
        "Baseline RAG": [0.97, 0.71, "4.97 s"],
        "CRAG": [0.86, 0.75, "12.19 s"],
    })
    st.caption("In the reported results, CRAG improved faithfulness, while baseline RAG had higher correctness and lower latency.")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Run the retrieval and evaluation pipeline without exposing a traceback
        # in the chat UI if a dependency or external service fails.
        try:
            response_stream, sources, trace = answer_query(prompt)
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        except Exception as exc:
            error_message = f"Could not process this question: {exc}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": "No sources.",
            })
            st.stop()
        
        # Post-process to separate thinking and answer
        thinking, clean_answer = split_response(full_response)
        message_placeholder.markdown(clean_answer)
        
        # Show thinking in expander if present
        if thinking:
            with st.expander("🧠 Model Reasoning (click to expand)"):
                st.markdown(thinking)
        
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
