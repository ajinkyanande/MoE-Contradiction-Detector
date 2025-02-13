import requests
import streamlit as st


# Streamlit Page Setup
st.set_page_config(page_title="TruthGuard: AI-Powered Text Verification", layout="wide")


# Color Mapping for Highlighting
GROUP_COLORS = [
    "#FF5733",
    "#33FF57",
    "#5733FF",
    "#F4C724",
    "#24F4F4",
    "#F424F4",
    "#F42424",
    "#24F424",
    "#2424F4",
    "#F4F424",
    "#F424F4",
    "#F4F4F4",
    "#242424",
]


# Backend API URL (Update if hosted elsewhere)
API_URL = "http://127.0.0.1:8000/detect/"


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


def highlight_text(text, groups, label):
    # Group sentences based on entailment or contradiction
    grouped_sentences = {}

    for sentence, group_id in groups.items():
        if group_id not in grouped_sentences:
            grouped_sentences[group_id] = []
        grouped_sentences[group_id].append(sentence)

    # Highlight sentences based on group
    highlighted_text = text

    for idx, sentences in grouped_sentences.items():
        color = GROUP_COLORS[idx % len(GROUP_COLORS)]
        for sentence in sentences:
            highlighted_text = highlighted_text.replace(
                sentence, f'<span style="background-color:{color};">{sentence}</span>'
            )

    return f"<p><b>{label}:</b></p><p>{highlighted_text}</p>"


# UI Header
st.title("TruthGuard: AI-Powered Fact & Consistency Checker")
st.write(
    "Compare and validate text sources for contradictions, inconsistencies, and factual alignment."
)


# User Input Section
st.subheader("🔍 Input Your Text for Analysis")
st.write(
    "Provide a reference text and a comparison text to analyze factual consistency and contradictions."
)


text1 = st.text_area("Reference Text (Ground Truth)", default_text1, height=200)
text2 = st.text_area("Comparison Text (To Be Verified)", default_text2, height=200)


if st.button("Analyze Text for Consistency"):
    if text1 and text2:
        # Display processing message and remove once results are ready
        with st.spinner("Analyzing text for consistency..."):
            # Fetch comparison results
            result = fetch_comparison(text1, text2)

        if result:
            entailments = result.get("entailment_groups", {})
            contradictions = result.get("contradiction_groups", {})

            # Display Analysis Results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🟢 Reference Text (Ground Truth)")
                st.markdown(
                    highlight_text(
                        text1, entailments, "Aligned / Consistent Information"
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_text(text1, contradictions, "Contradictory Information"),
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown("### 🔴 Comparison Text (Under Review)")
                st.markdown(
                    highlight_text(text2, entailments, "Correct / Consistent Claims"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_text(
                        text2, contradictions, "Incorrect / Contradictory Claims"
                    ),
                    unsafe_allow_html=True,
                )

        else:
            st.error(
                "❌ Error: No response from backend. Please check the API connection."
            )

    else:
        st.warning("⚠ Please enter text in both fields to proceed.")
