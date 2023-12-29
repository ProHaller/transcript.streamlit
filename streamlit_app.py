import os
from concurrent.futures import ThreadPoolExecutor
from logging import PlaceHolder
from tempfile import mkdtemp

import openai
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment


def display_readme():
    st.image("static/pointing.png", width=300)
    st.markdown(
        """
# 🤖💬 Welcome to Roland Tools

This application utilizes OpenAI's powerful models for audio transcription and text processing. To get started, please follow these simple steps:

### Mobile Users may need to open the side panel ↖️ 

## Using the App
### Transcription
- **Upload an Audio File**: Choose an audio file (mp3, wav, or m4a format).
- **Choose the Language**: Select the language of the audio file.
- **Optional Description**: Provide a brief description of the audio and the unusual vocabulary for better context and transcription.
- **Transcription Format**: Choose between plain text or subtitles (SRT format).
- **Transcribe**: Click the 'Transcribe' button to start the transcription process.

### Text Processing
- **Input Your Prompt**: Type or paste the text you want to process in the text area.
- **Choose a Model**: Select either GPT-4 or GPT-3.5 based on your preference.
- **Set the Temperature**: Adjust the slider to set the model's creativity.
- **Process Text**: Click the 'Process text' button to start.

## Download Results
- After transcription or text processing, you can download the results using the 'Download' button.

Enjoy your experience with Roland Tools! If you have any questions or feedback, please feel free to reach out.
    """
    )


# Check if the user has seen the README
if "readme_displayed" not in st.session_state:
    st.session_state["readme_displayed"] = False

if not st.session_state["readme_displayed"]:
    display_readme()
    st.session_state["readme_displayed"] = True


def transcription(file_path, language="en", prompt="", response_format="text"):
    client = OpenAI()

    # Open the file and pass the file handle directly
    with open(file_path, "rb") as file_handle:
        transcription_data = client.audio.transcriptions.create(
            model="whisper-1",
            file=file_handle,  # Pass the file handle
            language=language,
            prompt=prompt,
            response_format=response_format,
        )
    return transcription_data


def get_prompt_choice():
    prompt_options = {
        "None": "",
        "Summary": "Summaries the main points of the text into a concise report.",
        "Meeting minutes": "From the meeting transcript provided, create the meeting minutes.",
        "Make notes": "From the transcript provided, create a structured note in markdown.",
    }
    return st.selectbox(
        "Choose a prompt:",
        options=list(prompt_options.values()),
        format_func=lambda x: [
            key for key, value in prompt_options.items() if value == x
        ][0],
    )


# Function to segment the audio file
def segment_audio(uploaded_file, segment_duration=60000):  # Duration in milliseconds
    temp_dir = mkdtemp()  # Create a temporary directory
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio = AudioSegment.from_file(temp_file_path)
    segments = []
    for i in range(0, len(audio), segment_duration):
        segment = audio[i : i + segment_duration]
        segment_file_path = os.path.join(temp_dir, f"segment_{i//segment_duration}.mp3")
        segment.export(segment_file_path, format="mp3")
        segments.append(segment_file_path)
    return segments


# Function for parallel audio transcription
def parallel_transcribe_audio(
    file_paths, language, prompt, response_format, max_workers=10
):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                transcription, file_path, language, prompt, response_format
            ): i
            for i, file_path in enumerate(file_paths)
        }

    transcriptions = {}
    for future in futures:
        try:
            transcription_data = future.result()
            index = futures[future]
            transcriptions[index] = transcription_data
        except Exception as e:
            st.error(f"An error occurred during transcription: {e}")

    sorted_transcription_texts = [
        transcriptions[i]
        for i in sorted(transcriptions)
        if transcriptions[i] is not None
    ]
    full_transcript = " ".join(sorted_transcription_texts)
    return full_transcript


def openai_completion(
    input_text: str,
    system_prompt: str = "",
    format="text",
    model: str = "gpt-4-1106-preview",
    temperature: float = 0,
):
    client = OpenAI()
    completion_data = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        response_format={"type": format},
    )
    return completion_data.choices[0].message.content


# Initialize session state for transcription text
if "transcription_text" not in st.session_state:
    st.session_state["transcription_text"] = ""


# Function to display full language selector and return iso code.
def get_language_choice():
    language_options = {
        "English": "en",
        "Japanese": "ja",
        "French": "fr",
        "Thai": "th",
        "Arabic": "ar",
        "Chinese": "zh",
        "Korean": "ko",
    }
    return st.selectbox(
        "Choose a language:",
        options=list(language_options.values()),
        format_func=lambda x: [
            key for key, value in language_options.items() if value == x
        ][0],
    )


with st.sidebar:
    st.image("static/salute_cut.png", width=200)
    st.title("🤖💬 Roland Tools")
    if "OPENAI_API_KEY" in st.secrets:
        st.success(
            "The OpenAI credentials have been entered for you! You are all set!",
            icon="✅",
        )
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai.api_key = st.text_input("Enter OpenAI API token:", type="password")
        if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 51):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to uploading your audio file!", icon="👉")

    tab1, tab2 = st.tabs(["Transcription", "Text processing"])
    with tab1:
        st.header("Upload an audio file")
        uploaded_file = st.file_uploader(
            "Choose an audio file ⬇️",
            type=[
                "flac",
                "m4a",
                "mp3",
                "mp4",
                "mpeg",
                "mpga",
                "oga",
                "ogg",
                "wav",
                "webm",
            ],
        )
        if uploaded_file:
            st.audio(uploaded_file)
            language = get_language_choice()
            response_format = "srt" if st.toggle("Transcribe to subtitles") else "text"
            prompt = st.text_input(
                "Describe the audio (optional):",
                placeholder="This is a conversation between 2 people. Vocabulary: Tsunagaru, Roland Haller, Alice Ballé…",
                help="This can help the transcription to be more accurate by providing context and vocabulary.",
            )
        transcribe_button = st.button("Transcribe audio")
    with tab2:
        st.header("Process the Text:")
        completion_text = ""
        prepared_prompt = get_prompt_choice() or ""
        processing_prompt = st.text_area(
            f"Prompt: {prepared_prompt}",
        )
        model = st.radio(
            "Model",
            ["gpt-4-1106-preview", "gpt-3.5-turbo"],
            captions=["Best for most tasks", "Best for formatting"],
            horizontal=True,
        )
        temperature = st.slider(
            "Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1
        )
        process_button = st.button("Process text")

if transcribe_button:
    with st.spinner("Wait for it... our AI is flexing its muscles!"):
        st.image("static/writing.png", width=300)
        segment_paths = segment_audio(uploaded_file)
        st.session_state["transcription_text"] = parallel_transcribe_audio(
            segment_paths, language, prompt, response_format
        )
    st.success("Done!")
    st.balloons()

if process_button:
    with st.spinner("Just a moment... our AI is brainstorming!"):
        st.image("static/thinking.png", width=300)
        completion_text = openai_completion(
            input_text=processing_prompt
            + prepared_prompt
            + (
                st.session_state["transcription_text"]
                if st.session_state["transcription_text"]
                else ""
            ),
            system_prompt="",
            format="text",
            model=model,
            temperature=temperature,
        )
    st.success("Done!")
    st.balloons()

if st.session_state["transcription_text"]:
    transcription_download = st.download_button(
        label="Download transcription",
        data=st.session_state["transcription_text"],
        file_name=uploaded_file.name.rsplit(".", 1)[0]
        + "_transcription"
        + (".srt" if tab1.response_format is True else ".txt"),
    )
    "You can now process the text with the 'Text processing' tab."
    st.markdown("# Transcription:")
    st.write(st.session_state["transcription_text"])
if completion_text:
    "---"
    process_download = st.download_button(
        label="Download processed text",
        data=completion_text,
        file_name=(
            uploaded_file.name.rsplit(".", 1)[0] + "_processed" + ".txt"
            if uploaded_file
            else "Processed_text.txt"
        ),
    )
    st.markdown("# Processed Text:")
    st.write(completion_text)
    st.image("static/thumbsup.png", width=300)

