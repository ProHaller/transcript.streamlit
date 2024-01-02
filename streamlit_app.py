import gettext
import os
import time
from concurrent.futures import ThreadPoolExecutor
from tempfile import mkdtemp

import openai
import streamlit as st
import toml
from openai import OpenAI
from pydub import AudioSegment
from st_audiorec import st_audiorec

st.set_page_config(
    page_title="_(Roland'sTool)",
    page_icon=("🤖"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Globals
global _
_ = gettext.gettext

# Initialize session state for transcription text
if "readme_displayed" not in st.session_state:
    st.session_state["readme_displayed"] = False


def load_language(lang_code):
    global _  # Declare _ as global at the start of the function
    if lang_code == "en":
        _ = gettext.gettext  # Default to built-in gettext for English
    else:
        mo_file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "locales",
                lang_code,
                "LC_MESSAGES",
                "messages.mo",
            )
        )
        print(f"MO file path: {mo_file_path}")  # Debugging print

        try:
            with open(mo_file_path, "rb") as mo_file:
                # Load the .mo file directly
                localizator = gettext.GNUTranslations(mo_file)
                localizator.install()
                _ = (
                    localizator.gettext
                )  # Set the _ to the gettext function from localizator
                print(f"Language loaded: {lang_code}")  # Debugging print
        except FileNotFoundError as e:
            print(f"Exception loading MO file: {e}")
            st.error(f"Exception loading MO file: {e}")


def display_readme(lang_code="en"):
    load_language(lang_code)  # This will set the _ function correctly

    # Use the _() function directly on the strings to be translated
    welcome_text = _(
        ""
        "\n"
        "# Welcome to Roland Tools\n"
        "\n"
        "This app transcribe spoken words from any language then make useful notes"
        " from it.\n"
        "    "
    )

    st.image("static/transcription.svg", width=400, use_column_width="always")
    st.markdown(welcome_text)

    file_name = "README_JP.md" if lang_code == "ja" else "README.md"
    with open(file_name, "r") as file:
        readme_content = file.read()
    with st.expander(_("I need help!")):
        st.markdown(readme_content)


# Checkbox for language selection
if st.sidebar.checkbox("🇯🇵 日本語 🇯🇵", key="ja_check"):
    display_readme("ja")
else:
    display_readme("en")


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
    prompt_options = toml.load("prompts.toml")
    formatted_options = {
        key: value["description"] for key, value in prompt_options.items()
    }
    return st.selectbox(
        _("Choose prepared AI secretary intructions (Optional):"),
        key="prompt_box",
        options=formatted_options.values(),
        format_func=lambda x: next(k for k, v in formatted_options.items() if v == x),
    )


# Function to segment the audio file
def segment_audio(audio_file, segment_duration=60000):
    temp_dir = mkdtemp()
    file_name = "audio_segment.wav"  # A generic name for the audio segment
    temp_file_path = os.path.join(temp_dir, file_name)

    audio_file.seek(0)  # Ensure the file pointer is at the start
    with open(temp_file_path, "wb") as f:
        f.write(audio_file.read())

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
    file_paths,
    language,
    prompt,
    response_format,
    max_workers=10,
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
            st.error(_(f"An error occurred during transcription: {e}"))

    sorted_transcription_texts = [
        transcriptions[i]
        for i in sorted(transcriptions)
        if transcriptions[i] is not None
    ]
    full_transcript = " ".join(sorted_transcription_texts)
    return full_transcript


def openai_completion(
    input_text: str,
    system_prompt: str = "You are a helpful assistant that always answers in markdown.",
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

# Initialize session state for completion text
if "completion_text" not in st.session_state:
    st.session_state["completion_text"] = ""


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
        _("What is the language of the audio?"),
        options=list(language_options.values()),
        format_func=lambda x: [
            key for key, value in language_options.items() if value == x
        ][0],
        key="get_language_choice_selctor",
    )


with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "static/logo.png",
            width=100,
            use_column_width="always",
        )
    with col2:
        if "OPENAI_API_KEY" in st.secrets:
            st.write(_("The OpenAI credentials have been entered for you!"))
            st.write(_("You are all set!"))
            openai.api_key = st.secrets["OPENAI_API_KEY"]
        else:
            openai.api_key = st.text_input(
                _("Enter OpenAI API token:"), type="password"
            )
            if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 51):
                st.warning(_("Please enter your credentials!"), icon="⚠️")
            else:
                st.success(_("Proceed to uploading your audio file!"), icon="👉")
    "---"
    st.title(_("🎧🤖 Transcribe Tool"))
    tab1, tab2 = st.tabs([_("💽 Upload"), _("🎙️ Record")])
    with tab1:
        uploaded_file = st.file_uploader(
            "File Uploader",
            label_visibility="collapsed",
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
    with tab2:
        st.subheader("")
        recorded_file = st_audiorec()
    transcribe_button = None
    if uploaded_file or recorded_file:
        with st.form(key="transcribe_form", clear_on_submit=False, border=True):
            if uploaded_file:
                st.audio(uploaded_file)
            language = get_language_choice()
            response_format = "srt" if st.toggle("Transcribe to subtitles") else "text"
            prompt = st.text_area(
                _("Describe the audio (optional):"),
                placeholder=_(
                    "This is a conversation between 2 people. Vocabulary: Tsunagaru, Roland Haller, Alice Ballé…"
                ),
                help=_(
                    "This can help the transcription to be more accurate by providing context and vocabulary."
                ),
            )
            transcribe_button = st.form_submit_button(
                _("Transcribe audio"),
                type="primary",
                use_container_width=True,
            )
    else:
        transcribe_button_warning = st.button(
            _("Transcribe audio"),
            type="primary",
            use_container_width=True,
        )
        if transcribe_button_warning:
            st.error(_("Please upload a file to transcribe…"))
    "---"
    st.title(_("🤖📝 Secretary Tool"))
    prepared_prompt = get_prompt_choice() or ""
    with st.form(key="secretary_form", clear_on_submit=False, border=True):
        processing_prompt = st.text_area(
            _("AI secretary instructions:"),
            value=(f"{prepared_prompt}"),
            help=_(
                "Describe what you want your AI secretary to do with the transcribed text: make notes, a poem, a list of groceries, etc."
            ),
        )
        model = st.radio(
            "Model",
            ["gpt-4-1106-preview", "gpt-3.5-turbo"],
            captions=[_("Best for most tasks"), _("Best for formatting")],
            horizontal=True,
        )
        temperature = st.slider(
            _("Originality"),
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help=_(
                "This is the originality(temperature) of the openai model. 0 for a deterministic model always answering the same from the same input, 2 is fully free crazy AI completely detached from the input. 0.7 is the default."
            ),
        )
        process_button = st.form_submit_button(
            _("Process text"),
            type="primary",
            use_container_width=True,
        )
        is_festive = st.checkbox(_("I am feeling festive!"))

if transcribe_button:
    with st.spinner(_("Wait for it... our AI is listening!")):
        st.image(
            "static/writing.png",
            width=300,
        )

        # Check if there is recorded audio and no uploaded file
        if recorded_file is not None and uploaded_file is None:
            recorded_audio_path = os.path.join(
                mkdtemp(), "recorded_audio.wav"
            )  # Temporary file
            with open(recorded_audio_path, "wb") as f:
                f.write(recorded_file)  # Write the bytes directly
            audio_to_process = open(recorded_audio_path, "rb")
        else:
            # Use the uploaded file
            audio_to_process = uploaded_file

        segment_paths = segment_audio(audio_to_process)
        st.session_state["transcription_text"] = parallel_transcribe_audio(
            segment_paths, language, prompt, response_format
        )
    st.success("Done!")

if process_button:
    with st.spinner(_("Just a moment... our AI is thinking!")):
        st.image(
            "static/thinking.png",
            width=300,
        )
        st.session_state["completion_text"] = openai_completion(
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
        if is_festive:
            st.balloons()
    st.success("Done!")

# Check if the user has seen the README
if "readme_displayed" not in st.session_state:
    st.session_state["readme_displayed"] = False

if st.session_state["transcription_text"]:
    st.session_state["readme_displayed"] = True


if st.session_state["transcription_text"]:
    # Determine the file name based on whether the file was uploaded or recorded
    file_base_name = (
        uploaded_file.name.rsplit(".", 1)[0] if uploaded_file else "recorded_audio"
    )
    transcription_file_name = (
        file_base_name
        + "_transcription"
        + (".srt" if tab1.response_format is True else ".txt")
    )

    # Use this file name in the download button
    transcription_download = st.download_button(
        label="Download transcription",
        data=st.session_state["transcription_text"],
        file_name=transcription_file_name,
    )
    _("You can now process the text with the 'Text processing' tab.")
    st.markdown(_("# Transcription:"))
    st.write(st.session_state["transcription_text"])

if st.session_state["completion_text"]:
    "---"
    process_download = st.download_button(
        label=_("Download processed text"),
        data=st.session_state["completion_text"],
        file_name=(
            uploaded_file.name.rsplit(".", 1)[0] + "_processed" + ".txt"
            if uploaded_file
            else "Processed_text.txt"
        ),
    )
    st.markdown(_("# Post-processed Text:"))
    st.write(st.session_state["completion_text"])
    st.image("static/thumbsup.png", width=300)
