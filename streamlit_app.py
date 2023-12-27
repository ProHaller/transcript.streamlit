import openai
import streamlit as st
from openai import OpenAI


def transcription(file):
    client = OpenAI()
    transcription_data = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        language="en",
        prompt="",
        response_format="text",
    )
    return transcription_data

with st.sidebar:
    st.title('🤖💬 OpenAI Whisper')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        st.audio(uploaded_file)
        if transcribe := st.button("Transcribe"):
            transcription = transcription(uploaded_file)

st.markdown("# Hello this is _markdown_.")
st.write(transcription)
