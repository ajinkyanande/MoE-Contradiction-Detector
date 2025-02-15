import requests

import streamlit as st


# Backend API URL
API_URL = "http://localhost:8080/detect"

# Streamlit Page Setup
st.set_page_config(page_title="TruthGuard: AI-Powered Text Verification", layout="wide")

# Color Mapping for Highlighting
ENTAILMENT_GROUP_COLORS = [
    "rgba(255, 225, 25, 0.3)",  # Yellow
    "rgba(0, 130, 200, 0.3)",   # Blue
    "rgba(245, 130, 48, 0.3)",  # Orange
    "rgba(145, 30, 180, 0.3)",  # Purple
    "rgba(70, 240, 240, 0.3)",  # Cyan
    "rgba(240, 50, 230, 0.3)",  # Magenta
    "rgba(210, 245, 60, 0.3)",  # Lime
    "rgba(250, 190, 190, 0.3)", # Pink
    "rgba(0, 128, 128, 0.3)",   # Teal
    "rgba(60, 180, 75, 0.3)",   # Green
]
CONTRADICTION_GROUP_COLORS = [
    "rgba(230, 190, 255, 0.3)", # Lavender
    "rgba(170, 110, 40, 0.3)",  # Brown
    "rgba(255, 250, 200, 0.3)", # Beige
    "rgba(128, 0, 0, 0.3)",     # Maroon
    "rgba(170, 255, 195, 0.3)", # Mint
    "rgba(128, 128, 0, 0.3)",   # Olive
    "rgba(255, 215, 180, 0.3)", # Coral
    "rgba(0, 0, 128, 0.3)",     # Navy
    "rgba(128, 128, 128, 0.3)", # Grey
    "rgba(230, 25, 75, 0.3)",   # Red
]

default_text1 = (
    "Isaac Newton's theory of gravity was based on the idea that all masses attract each other "
    "with a force proportional to their mass and inversely proportional to the square of their distance. "
    "His equations successfully described planetary motion and were used for centuries in astronomy. "
    "However, in the early 20th century, Albert Einstein proposed a radically different idea. "
    "According to general relativity, gravity is not a force but rather the curvature of spacetime caused by mass. "
    "Large objects, such as the Sun, bend spacetime, affecting how planets move. "
    "This theory was confirmed when light from distant stars was observed bending around the Sun during a solar eclipse. "
    "Despite its success, general relativity does not fully explain gravity at the quantum level, "
    "leading to the search for a theory of quantum gravity."
)

default_text2 = (
    "Isaac Newton described gravity as a fundamental force acting at a distance, governed by an inverse-square law. "
    "While useful for everyday physics, Newtonian gravity does not accurately describe the behavior of light or extreme conditions such as black holes. "
    "Albert Einstein introduced general relativity, which describes gravity as the curvature of spacetime rather than a force. "
    "Experimental evidence, such as gravitational lensing, has confirmed Einstein's predictions. "
    "However, general relativity is incompatible with quantum mechanics. "
    "Physicists believe a theory of quantum gravity is necessary to unify gravity with the other fundamental forces. "
    "Some proposed models, such as string theory and loop quantum gravity, aim to bridge this gap."
)


def fetch_comparison(text1, text2):
    response = requests.post(API_URL, json={"text1": text1, "text2": text2})
    return response.json() if response.status_code == 200 else {}


def highlight_entailment_text(text, groups):
    highlighted_text = text
    for idx, (sentence, group_id) in enumerate(groups.items()):
        color = ENTAILMENT_GROUP_COLORS[group_id % len(ENTAILMENT_GROUP_COLORS)]
        highlighted_text = highlighted_text.replace(
            sentence, f'<span style="background-color:{color};">{sentence}</span>'
        )
    return f"<p>{highlighted_text}</p>"


def highlight_contradiction_text(text, groups):
    highlighted_text = text
    for idx, (sentence, group_id) in enumerate(groups.items()):
        color = CONTRADICTION_GROUP_COLORS[group_id % len(CONTRADICTION_GROUP_COLORS)]
        highlighted_text = highlighted_text.replace(
            sentence, f'<span style="background-color:{color};">{sentence}</span>'
        )
    return f"<p>{highlighted_text}</p>"


# Page Layout
st.title("🔍 TruthGuard: AI-Powered Fact & Consistency Checker")
st.write("Compare and validate text sources for contradictions, inconsistencies, and factual alignment.")

# Text Input Fields
st.markdown("---")
text1 = st.text_area("📖 Reference Text (Ground Truth)", default_text1, height=200)
text2 = st.text_area("📝 Comparison Text (To Be Verified)", default_text2, height=200)

st.markdown(
    """
    <style>
        .resizable-box {
            width: 100%;
            min-height: 100px;
            max-height: 400px;
            overflow: auto;
            resize: vertical;
            border: 1px solid #30363d;
            padding: 10px;
            background-color: #0d1117;
            color: #c9d1d9;
            border-radius: 5px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Analyze Text for Consistency"):
    if text1 and text2:
        # Fetch comparison results from the backend API
        result = fetch_comparison(text1, text2)

        if result:
            entailments = result.get("entailment_groups", {})
            contradictions = result.get("contradiction_groups", {})

            # How to read the output
            st.markdown('<div class="section-title">📚 How to Read the Output</div>', unsafe_allow_html=True)
            st.markdown("- Each group of contradictions, entailments, or neutral sentences is highlighted in the same color.")
            st.markdown("- Example: 'My name is A' vs. 'My name is B' vs. 'My name is C' are part of one contradiction group.")

            # Aligned / Consistent Information
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🟢 Aligned / Consistent Information</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Reference Text (Ground Truth)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="resizable-box">{highlight_entailment_text(text1, entailments)}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Comparison Text (Under Review)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="resizable-box">{highlight_entailment_text(text2, entailments)}</div>', unsafe_allow_html=True)

            # Incorrect / Contradictory Claims
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔴 Incorrect / Contradictory Claims</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Reference Text (Ground Truth)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="resizable-box">{highlight_contradiction_text(text1, contradictions)}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Comparison Text (Under Review)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="resizable-box">{highlight_contradiction_text(text2, contradictions)}</div>', unsafe_allow_html=True)

        else:
            st.error("❌ No response from backend. Please check the API connection.")

    else:
        st.warning("⚠ Please enter text in both fields to proceed.")

# Applications Section (Moved below)
st.markdown("---")
st.markdown("#### 🚀 How Can This Tool Be Used?")
st.write("TruthGuard can be applied across multiple industries to detect contradictions and ensure factual consistency.")

applications = [
    ("💬 LLM Response Verification", "Ensure AI-generated responses do not contain hallucinations or factual errors."),
    ("🔎 Fact-Checking & Misinformation Detection", "Compare news articles or AI-generated content to verify truthfulness."),
    ("📰 Journalism & Editorial Fact-Checking", "Verify news articles, editorials, and opinion pieces for factual alignment."),
    ("📚 Academic & Research Cross-Validation", "Validate research papers, summaries, and thesis content for logical consistency."),
    ("📜 Legal Document Consistency Analysis", "Check contracts, terms, and policies for contradictions."),
    ("🗂️ Medical Report & Diagnosis Consistency", "Identify contradictions in medical records and AI-assisted diagnoses."),
]

for title, description in applications:
    st.markdown(f"##### {title}")
    st.write(description)

# Contact Information (Bottom-most section)
st.markdown("---")
st.markdown("#### 👨‍💻 Developed by: **Ajinkya Nande**")

# Adding LinkedIn and GitHub Logos with Links
linkedin_logo = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
github_logo = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
folder_logo = "https://img.icons8.com/ios/452/opened-folder.png"

st.markdown(
    f'<a href="https://www.linkedin.com/in/ajinkyanande/" target="_blank">'
    f'<img src="{linkedin_logo}" width="30" style="vertical-align: middle; margin-right: 10px;">'
    f'LinkedIn</a>',
    unsafe_allow_html=True
)

st.markdown(
    f'<a href="https://github.com/ajinkyanande" target="_blank">'
    f'<img src="{github_logo}" width="30" style="vertical-align: middle; margin-right: 10px;">'
    f'GitHub</a>',
    unsafe_allow_html=True
)

st.markdown(
    f'<a href="https://github.com/ajinkyanande/MoE-Contradiction-Detector" target="_blank">'
    f'<img src="{folder_logo}" width="30" style="vertical-align: middle; margin-right: 10px;">'
    f'GitHub/MoE-Contradiction-Detector</a>',
    unsafe_allow_html=True
)
