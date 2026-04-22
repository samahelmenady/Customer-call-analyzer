import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

st.set_page_config(page_title="AI Customer Call Analyzer", layout="wide")

st.title("AI Customer Call Analyzer")
st.write("Upload a customer support call recording and get an instant AI analysis using Gemini.")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav", "m4a", "mpeg"]
)

def analyze_audio_with_gemini(file_path):
    uploaded = client.files.upload(file=file_path)

    prompt = """
You are an AI assistant specialized in analyzing customer support calls.

Analyze the audio and return ONLY valid JSON in this exact format:

{
  "transcript": "...",
  "summary": "...",
  "main_issue": "...",
  "sentiment": "positive/neutral/negative",
  "urgency": "low/medium/high",
  "resolved_status": "resolved/not resolved/partially resolved",
  "next_action": "...",
  "customer_tone": "...",
  "agent_performance": "..."
}

Return JSON only.
Do not use markdown.
Do not add any explanation outside the JSON.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[prompt, uploaded]
    )

    text = response.text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("Model did not return valid JSON.")

    json_text = text[start:end]
    return json.loads(json_text)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Analyze Call"):
        temp_path = None
        try:
            with st.spinner("Uploading and analyzing audio..."):
                suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ".mp3"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = tmp.name

                result = analyze_audio_with_gemini(temp_path)

            st.subheader("Transcript")
            st.write(result["transcript"])

            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Summary:** {result['summary']}")
                st.markdown(f"**Main Issue:** {result['main_issue']}")
                st.markdown(f"**Sentiment:** {result['sentiment']}")
                st.markdown(f"**Urgency:** {result['urgency']}")

            with col2:
                st.markdown(f"**Resolved Status:** {result['resolved_status']}")
                st.markdown(f"**Next Action:** {result['next_action']}")
                st.markdown(f"**Customer Tone:** {result['customer_tone']}")
                st.markdown(f"**Agent Performance:** {result['agent_performance']}")

            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)