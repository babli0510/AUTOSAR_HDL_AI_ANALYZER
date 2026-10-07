import streamlit as st
from pathlib import Path
import sys
import tempfile
import shutil


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT / "src"))


# ============================================================
# IMPORT RAG SYSTEM
# ============================================================

from rag.answer_generator import RAGAnswerGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AUTOSAR HLD AI Analyzer",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
    }

    .source-box {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 AUTOSAR HLD AI Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered AUTOSAR High-Level Design Document Analysis Assistant'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 HLD Document")

    uploaded_file = st.file_uploader(
        "Upload AUTOSAR HLD PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown("### System")

    st.success("RAG Pipeline Active")

    st.caption(
        "PDF → Chunking → Embeddings → "
        "ChromaDB → Retrieval → Qwen2.5"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:

    st.session_state.rag = None


if "document_name" not in st.session_state:

    st.session_state.document_name = None


# ============================================================
# INITIALIZE RAG
# ============================================================

if st.session_state.rag is None:

    with st.spinner(
        "Initializing AUTOSAR HLD analysis engine..."
    ):

        try:

            st.session_state.rag = RAGAnswerGenerator(
                top_k=3
            )

        except Exception as error:

            st.error(
                f"Failed to initialize RAG system: {error}"
            )

            st.stop()


# ============================================================
# DOCUMENT SECTION
# ============================================================

st.header("📑 Document Analysis")


if uploaded_file:

    st.session_state.document_name = (
        uploaded_file.name
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Document",
            uploaded_file.name
        )

    with col2:

        st.metric(
            "Type",
            "AUTOSAR HLD PDF"
        )

    with col3:

        st.metric(
            "Analysis Mode",
            "RAG"
        )

    st.success(
        "HLD document uploaded successfully."
    )

else:

    st.info(
        "Upload an AUTOSAR HLD PDF from the sidebar "
        "to begin analysis."
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.header("💬 Ask the HLD Assistant")


question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: Which component provides "
        "VehicleState to ClimateController?"
    ),
    height=100
)


# ============================================================
# SAMPLE QUESTIONS
# ============================================================

st.markdown("### 🔍 Example Questions")

example_questions = [

    "Which component provides VehicleState to ClimateController?",

    "What interfaces are used by ClimateController?",

    "What signals are provided by SensorManager?",

    "What is the responsibility of FanController?",

    "What is the dependency between VehicleStateManager and ClimateController?",

    "What is the functional flow of the Vehicle Climate Control ECU?"
]


selected_question = st.selectbox(
    "Select a sample question",
    ["-- Select --"] + example_questions
)


if selected_question != "-- Select --":

    question = selected_question


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze HLD",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANSWER GENERATION
# ============================================================

if analyze:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    with st.spinner(
        "Retrieving HLD evidence and generating answer..."
    ):

        try:

            result = st.session_state.rag.generate_answer(
                question
            )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

            st.stop()


    # ========================================================
    # ANSWER
    # ========================================================

    st.header("🤖 Analysis Result")

    st.markdown(
        '<div class="answer-box">',
        unsafe_allow_html=True
    )

    st.subheader("Answer")

    st.write(
        result.get(
            "answer",
            "No answer generated."
        )
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader("📚 Supporting HLD Evidence")


    sources = result.get(
        "sources",
        []
    )


    if sources:

        for index, source in enumerate(
            sources,
            start=1
        ):

            page = source.get(
                "page",
                "Unknown"
            )

            section = source.get(
                "section",
                "Unknown"
            )

            chunk = source.get(
                "chunk_id",
                "Unknown"
            )

            distance = source.get(
                "distance"
            )


            with st.container(
                border=True
            ):

                st.markdown(
                    f"**Source {index}**"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"📄 Page: **{page}**"
                    )

                with col2:

                    st.write(
                        f"📌 Section: **{section}**"
                    )

                with col3:

                    st.write(
                        f"🔹 Chunk: **{chunk}**"
                    )

                if distance is not None:

                    st.caption(
                        f"Retrieval distance: {distance:.4f}"
                    )


    else:

        st.warning(
            "No supporting evidence was retrieved."
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if sources:

        st.subheader("🎯 Evidence Confidence")

        st.success(
            "Evidence retrieved from the HLD document."
        )

    else:

        st.warning(
            "Low confidence — no supporting evidence found."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AUTOSAR HLD AI Analyzer | "
    "Retrieval-Augmented Generation (RAG) | "
    "Local LLM"
)