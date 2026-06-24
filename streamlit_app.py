"""Streamlit chat UI for the Production RAG API.

Run with:
    streamlit run streamlit_app.py

Requires the FastAPI backend to be running:
    uvicorn app.main:app --reload --port 8000
"""

import json

import httpx
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []       # [{role, content, citations}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None   # persists across turns for memory
if "namespace" not in st.session_state:
    st.session_state.namespace = "default"
if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_health(api_url: str) -> bool:
    """Return True if the API is reachable."""
    try:
        r = httpx.get(f"{api_url}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def do_ingest(api_url: str, source: str, namespace: str) -> dict:
    """Call POST /ingest and return the response dict or an error dict."""
    try:
        r = httpx.post(
            f"{api_url}/ingest",
            json={"source": source, "namespace": namespace},
            timeout=120.0,
        )
        if r.status_code == 200:
            return {"ok": True, **r.json()}
        return {"ok": False, "detail": r.json().get("detail", r.text)}
    except httpx.ConnectError:
        return {"ok": False, "detail": "Cannot reach the API. Is uvicorn running?"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def stream_query(api_url: str, question: str, namespace: str, session_id: str | None):
    """Generator that yields token strings from POST /query/stream.

    Stores the done-event metadata (citations, session_id, answer) into
    st.session_state so the caller can read it after the generator is exhausted.
    """
    payload: dict = {"question": question, "namespace": namespace}
    if session_id:
        payload["session_id"] = session_id

    st.session_state["_stream_meta"] = {}

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{api_url}/query/stream", json=payload) as resp:
                if resp.status_code != 200:
                    yield f"⚠️ API error {resp.status_code}: {resp.text}"
                    return
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if event["type"] == "token":
                        yield event["content"]
                    elif event["type"] == "done":
                        st.session_state["_stream_meta"] = {
                            "citations": event.get("citations", []),
                            "session_id": event.get("session_id"),
                            "answer": event.get("answer", ""),
                        }
                    elif event["type"] == "error":
                        detail = event.get("detail", "Unknown error")
                        if "No documents ingested" in detail:
                            yield "\n\n⚠️ No documents in this namespace yet. Use the sidebar to ingest some."
                        else:
                            yield f"\n\n⚠️ {detail}"

    except httpx.ConnectError:
        yield "⚠️ Cannot reach the API. Make sure `uvicorn app.main:app --reload --port 8000` is running."
    except Exception as exc:
        yield f"⚠️ Unexpected error: {exc}"


def render_citations(citations: list[dict]) -> None:
    """Render a citations expander for an assistant message."""
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} source{'s' if len(citations) != 1 else ''}"):
        for cit in citations:
            page_str = f" · page {cit['page']}" if cit.get("page") else ""
            short_id = cit["chunk_id"][:8] + "…"
            st.markdown(f"**{cit['source']}**{page_str}  \n`chunk: {short_id}`")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Ask My Docs")
    st.caption("Production RAG · Hybrid retrieval · Citations")

    st.divider()

    # API config + health indicator
    api_url = st.text_input("API URL", value="http://localhost:8000", key="api_url")
    healthy = check_health(api_url)
    if healthy:
        st.success("API is online", icon="🟢")
    else:
        st.error("API is offline", icon="🔴")

    st.divider()

    # Namespace selector
    st.subheader("🗂 Namespace")
    st.caption("Each namespace is an isolated knowledge base (like a NotebookLM notebook).")
    new_namespace = st.text_input(
        "Knowledge base name",
        value=st.session_state.namespace,
        placeholder="e.g. my-project, research-2024",
    )
    if new_namespace and new_namespace != st.session_state.namespace:
        st.session_state.namespace = new_namespace
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.ingest_result = None
        st.rerun()

    st.divider()

    # Ingest documents
    st.subheader("📥 Ingest Documents")
    source_input = st.text_input(
        "URL or file path",
        placeholder="https://... or /path/to/file.pdf",
        key="source_input",
    )
    if st.button("Ingest", type="primary", use_container_width=True, disabled=not healthy):
        if source_input.strip():
            with st.spinner(f"Ingesting into `{st.session_state.namespace}`…"):
                st.session_state.ingest_result = do_ingest(
                    api_url, source_input.strip(), st.session_state.namespace
                )
        else:
            st.warning("Enter a URL or file path first.")

    if st.session_state.ingest_result:
        result = st.session_state.ingest_result
        if result["ok"]:
            st.success(f"✅ {result['chunks_ingested']} chunks ingested")
        else:
            st.error(f"❌ {result['detail']}")

    st.divider()

    # Conversation controls
    st.subheader("💬 Conversation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()
    with col2:
        st.caption(
            f"Session  \n`{st.session_state.session_id[:8] + '…' if st.session_state.session_id else 'none'}`"
        )


# ── Main chat area ────────────────────────────────────────────────────────────

namespace_label = st.session_state.namespace
st.markdown(f"## 💬 Chat &nbsp;·&nbsp; `{namespace_label}`")

if not healthy:
    st.info(
        "The API is offline. Start it with:\n\n"
        "```bash\nuvicorn app.main:app --reload --port 8000\n```",
        icon="ℹ️",
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations(msg.get("citations", []))

# Chat input
if prompt := st.chat_input(
    "Ask a question…",
    disabled=not healthy,
):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            stream_query(api_url, prompt, st.session_state.namespace, st.session_state.session_id)
        )

        # Read metadata captured by stream_query
        meta = st.session_state.get("_stream_meta", {})
        citations = meta.get("citations", [])

        if meta.get("session_id"):
            st.session_state.session_id = meta["session_id"]

        render_citations(citations)

    # Persist to message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response or meta.get("answer", ""),
        "citations": citations,
    })
