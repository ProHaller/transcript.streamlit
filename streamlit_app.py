import os
from logging import PlaceHolder

import openai
import streamlit as st
from openai import OpenAI


def transcription(file, language="en", prompt="", response_format="text"):
    client = OpenAI()
    transcription_data = client.audio.transcriptions.create(
        model="whisper-1",
        file=file,
        language=language,
        prompt=prompt,
        response_format=response_format,
    )
    return transcription_data

with st.sidebar:
    st.title('🤖💬 OpenAI Whisper')
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('You are all set!', icon='✅')
        openai.api_key = st.secrets['OPENAI_API_KEY']
    else:
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    uploaded_file = st.file_uploader("Choose a file ⬇️")
    transcription_text = ""
    if uploaded_file:
        st.audio(uploaded_file)
        language_options = {
            "English": "en",
            "Japanese": "ja",
            "French": "fr",
            "Thai": "th",
            "Arabic": "ar",
            "Korean": "ko",
        }
        language = st.selectbox(
            "Choose a language:",
            options=list(language_options.values()),
            format_func=lambda x: [key for key, value in language_options.items() if value == x][0]
        )
        response_format = "srt" if st.toggle("Transcribe to subtitles") else "text"
        prompt = st.text_input("Describe the audio (optional):",placeholder="Tsunagaru, Roland Haller, Alice Ballé…")
        if st.button("Transcribe"):
            transcription_text = transcription(uploaded_file, language, prompt, response_format)
        if transcription_text:
            st.download_button(
                label="Download transcription",
                data=transcription_text,
                file_name=uploaded_file.name.rsplit('.', 1)[0] + '_transcription' + ".srt" if response_format else ".txt",
            )

if transcription_text:
    st.markdown("# Transcription:")
    st.write(transcription_text)
