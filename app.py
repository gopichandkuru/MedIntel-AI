
import requests
import streamlit as st
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================

API_URL = "http://localhost:8000/api/chat"

st.set_page_config(
    page_title="MedIntel-AI | Medical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL UI STYLING
# =====================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background: #F5F8FC;
    color: #172B4D;
}

/* Main content width */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #102A43;
    border-right: 1px solid #243F56;
}

[data-testid="stSidebar"] * {
    color: #EAF2F8 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #36536B;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background: #1B3A55;
    color: white !important;
    border: 1px solid #42617A;
    border-radius: 9px;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: #28516F;
    border-color: #67C5D8;
}

/* Main heading */
.brand-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #123B56;
    margin-bottom: 2px;
}

.brand-subtitle {
    color: #62778A;
    font-size: 15px;
    margin-bottom: 22px;
}

/* Status badge */
.status-pill {
    display: inline-block;
    background: #E5F5EE;
    color: #16764A;
    border: 1px solid #B8E5CE;
    border-radius: 20px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 700;
}

/* Disclaimer */
.disclaimer {
    background: #FFF9EA;
    border: 1px solid #F3D99B;
    border-left: 4px solid #D99A20;
    border-radius: 9px;
    padding: 14px 17px;
    color: #684D18;
    font-size: 13px;
    line-height: 1.6;
    margin: 16px 0 25px 0;
}

/* Section heading */
.section-heading {
    color: #173B56;
    font-size: 19px;
    font-weight: 700;
    margin: 20px 0 12px 0;
}

/* Welcome card */
.welcome-card {
    background: white;
    border: 1px solid #E1EAF1;
    border-radius: 15px;
    padding: 23px;
    margin: 12px 0 18px 0;
    box-shadow: 0 3px 12px rgba(20, 50, 75, 0.04);
}

.welcome-title {
    color: #173B56;
    font-size: 21px;
    font-weight: 750;
}

.welcome-text {
    color: #65788B;
    font-size: 14px;
    line-height: 1.6;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E3EBF2;
    border-radius: 14px;
    padding: 15px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(20, 50, 75, 0.035);
}

/* Chat message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #243B53 !important;
    line-height: 1.75;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: white;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #173B56 !important;
    border: 1px solid #D5E1EA !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8193A5 !important;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    border: 1px solid #DCE6EF;
    background: white;
    color: #23445D;
    font-weight: 600;
    min-height: 43px;
    transition: 0.2s ease;
}

.stButton button:hover {
    border-color: #1685A0;
    color: #126A83;
    background: #F0FAFC;
}

/* Expanders / sources */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E1EAF1;
    border-radius: 11px;
}

[data-testid="stExpander"] summary {
    color: #176B87 !important;
    font-weight: 650;
}

/* General text */
.stMarkdown p {
    color: #344B60;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #8797A6;
    font-size: 12px;
    padding-top: 25px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 18px 0;">
            <div style="font-size: 27px; font-weight: 800;">
                🩺 MedIntel-AI
            </div>
            <div style="font-size: 12px; color: #B5C9D9;">
                MEDICAL INTELLIGENCE PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Workspace")

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.markdown("### Platform overview")

    st.markdown("""
    **🧠 Intelligence**

    Local LLM · Ollama

    **📚 Knowledge base**

    WHO medical documents

    **🔎 Retrieval**

    Retrieval-Augmented Generation

    **⚙️ Backend**

    FastAPI
    """)

    st.divider()

    st.markdown("### About MedIntel-AI")

    st.caption(
        "A source-based medical knowledge assistant "
        "designed to provide educational information "
        "with traceable document references."
    )

    st.divider()

    st.markdown("### Privacy")

    st.caption(
        "Questions are processed through your configured "
        "local application backend."
    )

    st.markdown(
        """
        <div style="padding-top: 25px; color: #B5C9D9;
                    font-size: 11px;">
            MEDINTEL-AI · EDUCATIONAL PLATFORM
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# HEADER
# =====================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        '<div class="brand-title">MedIntel-AI 🩺</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">'
        'Medical knowledge. Evidence-based answers. '
        'Transparent sources.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<div style="text-align:right; padding-top:12px;">'
        '<span class="status-pill">● AI ASSISTANT</span>'
        '</div>',
        unsafe_allow_html=True
    )


# =====================================================
# MEDICAL DISCLAIMER
# =====================================================

st.markdown("""
<div class="disclaimer">
    <strong>⚕️ Medical information disclaimer</strong><br>
    MedIntel-AI provides educational information based on available
    medical documents. It is not a diagnostic tool and does not replace
    professional medical advice. Consult a qualified healthcare
    professional for personal medical concerns.
</div>
""", unsafe_allow_html=True)


# =====================================================
# WELCOME SECTION
# =====================================================

if not st.session_state.messages:

    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">
            Welcome to MedIntel-AI
        </div>
        <div class="welcome-text">
            Ask a medical knowledge question. The system retrieves
            relevant information from its available medical documents
            and presents an answer with source references.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Explore medical topics</div>',
        unsafe_allow_html=True
    )

    suggestions = [
        "What are the symptoms of hypoglycemia?",
        "What are the symptoms of diabetes?",
        "What lifestyle changes help manage diabetes?"
    ]

    col1, col2, col3 = st.columns(3)

    for column, suggestion in zip(
        [col1, col2, col3], suggestions
    ):
        with column:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


# =====================================================
# SOURCE DISPLAY
# =====================================================

def display_sources(sources):

    if not sources:
        st.caption("No document sources were returned.")
        return

    with st.expander(
        f"📚 Evidence and document references ({len(sources)})"
    ):

        for index, source in enumerate(sources, start=1):

            document = source.get("document", "Unknown document")
            page = source.get("page", "N/A")
            extraction = source.get("extraction_method")
            similarity = source.get("similarity")

            st.markdown(f"**Source {index}: {document}**")
            st.write(f"Document page: {page}")

            if extraction:
                st.caption(f"Extraction method: {extraction}")

            if similarity is not None:
                st.caption(
                    f"Retrieval similarity: {similarity}"
                )

            if index < len(sources):
                st.divider()


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            display_sources(message.get("sources", []))

            for warning in message.get("warnings", []):
                st.warning(warning)


# =====================================================
# CHAT INPUT AND API CONNECTION
# =====================================================

question = st.chat_input(
    "Ask a medical knowledge question..."
)

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

        with st.spinner(
            "Searching medical knowledge and preparing your answer..."
        ):

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
                    "No answer was returned."
                )

                sources = data.get("sources", [])
                warnings = data.get("warnings", [])

                st.markdown(answer)

                display_sources(sources)

                for warning in warnings:
                    st.warning(warning)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "warnings": warnings
                })

            except requests.exceptions.ConnectionError:

                st.error(
                    "Unable to connect to the FastAPI backend. "
                    "Check whether the backend is running on port 8000."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request took too long. Please try again."
                )

            except requests.exceptions.HTTPError as error:

                st.error(f"Backend error: {error}")

            except requests.exceptions.RequestException as error:

                st.error(f"Request failed: {error}")

            except Exception as error:

                st.error(f"Unexpected error: {error}")


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer-text">
    MedIntel-AI · Medical Knowledge Assistant<br>
    Built for educational and research purposes
</div>
""", unsafe_allow_html=True)
