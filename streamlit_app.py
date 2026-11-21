import requests
import streamlit as st
from src.reporting.pdf_generator import generate_pdf
# ==================================================
# CONFIG
# ==================================================

API_URL = "http://localhost:8000/api/chat"

st.set_page_config(
    page_title="MedIntel-AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# STYLING
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F6F8FB;
    color: #182E42;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer {
    visibility: hidden;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #102A43;
    border-right: 1px solid #29465F;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.1rem;
}

[data-testid="stSidebar"] * {
    color: #EAF2F8;
}

[data-testid="stSidebar"] hr {
    border-color: #35536B;
}

[data-testid="stSidebar"] .stButton button {
    background: #1C3B55;
    border: 1px solid #42617A;
    color: white;
    border-radius: 10px;
    min-height: 43px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #28516F;
    border-color: #55B7C8;
}

/* Brand */
.brand {
    font-size: 27px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.brand-small {
    color: #AFC4D5;
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 4px;
}

/* Main header */
.header-title {
    color: #153B55;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.3px;
    line-height: 1.3;
}

.header-subtitle {
    color: #718397;
    font-size: 14px;
    margin-top: 5px;
}

.status {
    display: inline-block;
    background: #E7F6EF;
    color: #178052;
    padding: 7px 12px;
    border-radius: 30px;
    border: 1px solid #C7E9D7;
    font-size: 11px;
    font-weight: 700;
}

/* Disclaimer */
.disclaimer {
    background: #FFF9EB;
    border: 1px solid #F2DEAC;
    border-left: 4px solid #D89A25;
    padding: 13px 16px;
    border-radius: 10px;
    color: #76551C;
    font-size: 12px;
    line-height: 1.7;
    margin: 20px 0;
}

/* Welcome panel */
.welcome {
    background: linear-gradient(120deg, #EAF5FA, #F5FAFC);
    border: 1px solid #D9EAF1;
    border-radius: 17px;
    padding: 28px;
    margin: 15px 0 22px 0;
}

.welcome h2 {
    color: #153B55;
    font-size: 24px;
    margin: 0 0 8px 0;
}

.welcome p {
    color: #627B8E;
    font-size: 14px;
    line-height: 1.7;
    margin: 0;
}

/* Section headings */
.section-title {
    color: #183B54;
    font-size: 17px;
    font-weight: 700;
    margin: 22px 0 12px 0;
}

/* Suggestion cards */
div.stButton > button {
    border-radius: 11px;
    border: 1px solid #DCE6EE;
    background: #FFFFFF;
    color: #29465E;
    font-weight: 600;
    min-height: 65px;
    padding: 10px 13px;
    text-align: left;
    transition: all .2s ease;
}

div.stButton > button:hover {
    border-color: #16849B;
    background: #F0FAFC;
    color: #126C82;
}

/* Chat cards */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E2EAF0;
    border-radius: 15px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(20, 50, 75, .035);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #29445B !important;
    font-size: 14px;
    line-height: 1.8;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #FFFFFF;
    border: 1px solid #DCE6EE;
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #183B54 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #91A1AF !important;
}

/* Sources */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DFE8EF;
    border-radius: 11px;
}

[data-testid="stExpander"] summary {
    color: #176B87 !important;
    font-weight: 700;
}

/* Footer */
.footer {
    text-align: center;
    color: #91A0AD;
    font-size: 11px;
    padding: 25px 0;
}

/* Responsive */
@media(max-width: 768px) {
    .header-title {
        font-size: 26px;
    }

    .welcome {
        padding: 19px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:8px 0 20px 0;">
        <div class="brand">🩺 MedIntel-AI</div>
        <div class="brand-small">MEDICAL INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("#### WORKSPACE")

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown("#### PLATFORM")

    st.markdown("""
    **🧠 AI Intelligence**

    <span style="color:#AFC4D5;font-size:13px;">
    Local LLM · Ollama
    </span>

    <br><br>

    **📚 Medical Knowledge**

    <span style="color:#AFC4D5;font-size:13px;">
    WHO medical documents
    </span>

    <br><br>

    **🔎 Retrieval Engine**

    <span style="color:#AFC4D5;font-size:13px;">
    Retrieval-Augmented Generation
    </span>

    <br><br>

    **⚙️ API Infrastructure**

    <span style="color:#AFC4D5;font-size:13px;">
    FastAPI backend
    </span>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("#### ABOUT")

    st.markdown("""
    <span style="color:#AFC4D5;font-size:12px;line-height:1.8;">
    An educational medical knowledge assistant that connects
    answers with their supporting document references.
    </span>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("""
    <div style="color:#AFC4D5;font-size:10px;margin-top:20px;">
    MEDINTEL-AI · EDUCATIONAL PLATFORM
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# MAIN HEADER
# ==================================================

left, right = st.columns([4, 1])

with left:
    st.markdown(
        '<div class="header-title">MedIntel-AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="header-subtitle">'
        'Medical knowledge. Evidence-based answers. Transparent sources.'
        '</div>',
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        '<div style="text-align:right;padding-top:12px;">'
        '<span class="status">● AI ASSISTANT</span></div>',
        unsafe_allow_html=True
    )


# ==================================================
# DISCLAIMER
# ==================================================

st.markdown("""
<div class="disclaimer">
    <strong>⚕ Medical information disclaimer</strong><br>
    MedIntel-AI provides educational information based on available
    medical documents. It is not a diagnostic tool and does not replace
    professional medical advice. Consult a qualified healthcare
    professional for personal medical concerns.
</div>
""", unsafe_allow_html=True)


# ==================================================
# WELCOME + SUGGESTIONS
# ==================================================

if not st.session_state.messages:

    st.markdown("""
    <div class="welcome">
        <h2>Welcome to your medical knowledge workspace.</h2>
        <p>
        Explore medical topics using our document-based AI assistant.
        Ask a question and review the evidence behind the answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Suggested questions</div>',
        unsafe_allow_html=True
    )

    suggestions = [
        "What are the symptoms of hypoglycemia?",
        "What is diabetes?",
        "What lifestyle changes help manage diabetes?"
    ]

    c1, c2, c3 = st.columns(3)

    for col, suggestion in zip([c1, c2, c3], suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


# ==================================================
# SOURCE CARDS
# ==================================================

def show_sources(sources):

    if not sources:
        return

    with st.expander(
        f"📚 Evidence and document references · {len(sources)}"
    ):

        for i, source in enumerate(sources, start=1):

            document = source.get("document", "Unknown document")
            page = source.get("page", "N/A")
            extraction = source.get("extraction_method")
            similarity = source.get("similarity")

            st.markdown(f"**SOURCE {i}**")

            st.markdown(f"📄 {document}")
            st.caption(f"Document page: {page}")

            if extraction:
                st.caption(f"Extraction: {extraction}")

            if similarity is not None:
                st.caption(
                    f"Retrieval similarity: {similarity}"
                )

            if i != len(sources):
                st.divider()


# ==================================================
# CHAT HISTORY
# ==================================================

for message_index, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            show_sources(message.get("sources", []))

            for warning in message.get("warnings", []):
                st.warning(warning)

            # PDF DOWNLOAD
            pdf_bytes = message.get("pdf_bytes")

            if pdf_bytes:
                st.download_button(
                    label="📄 Download Medical Report (PDF)",
                    data=pdf_bytes,
                    file_name="MedIntel_Medical_Report.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{message_index}"
                )

# ==================================================
# CHAT REQUEST
# ==================================================

question = st.chat_input("Ask your medical question...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching medical documents..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=180
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer was returned by the backend."
                )

                sources = data.get("sources", [])
                warnings = data.get("warnings", [])

                st.markdown(answer)

                show_sources(sources)

                for warning in warnings:
                    st.warning(warning)

                # ==========================================
                # ==========================================
                # GENERATE MEDICAL PDF REPORT
                # ==========================================

                pdf_bytes = None

                try:
                    # Create a concise report without duplicate findings
                    # or internal retrieval metadata tables.
                    source_ids = [
                        source["id"]
                        for source in sources
                        if isinstance(source, dict) and source.get("id")
                    ]

                    report = {
                        "title": "Medical Research Report",
                        "question": question,
                        "answer": "",
                        "summary": "",
                        "sections": [{
                            "heading": "Evidence-based answer",
                            "content": answer,
                            "citations": source_ids
                        }],
                        "key_points": [],
                        "tables": [],
                        "charts": [],
                        "flowcharts": [],
                        "visuals": [],
                        "sources": sources,
                        "warnings": [],
                        "limitations": [
                            "Educational information only. "
                            "Not a diagnosis or medical advice."
                        ]
                    }

                    pdf_path = generate_pdf(report)

                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()

                except Exception as error:
                    st.warning(f"PDF generation failed: {error}")

                # Show the download button immediately.
                if pdf_bytes:
                    st.download_button(
                        label="📄 Download Medical Report (PDF)",
                        data=pdf_bytes,
                        file_name="MedIntel_Medical_Report.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_current_{len(st.session_state.messages)}"
                    )

                # Save assistant response and PDF in chat history.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "warnings": warnings,
                    "pdf_bytes": pdf_bytes
                })

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Check that the backend is running on port 8000."
                )

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except requests.exceptions.HTTPError as error:
                st.error(f"Backend HTTP error: {error}")

            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")

            except Exception as error:
                st.error(f"Unexpected error: {error}")


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    MedIntel-AI · Medical Knowledge Assistant<br>
    Educational and research purposes only
</div>
""", unsafe_allow_html=True)
