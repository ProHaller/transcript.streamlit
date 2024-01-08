import gettext
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Literal
import io
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

# Initializing session states
# This is a best practice to declare all session states upfront for clarity
if "password_ok" not in st.session_state:
    st.session_state["password_ok"] = None
if "openai_key" not in st.session_state:
    st.session_state["openai_key"] = (
        st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
    )
if "readme_displayed" not in st.session_state:
    st.session_state["readme_displayed"] = False
if "subtitles" not in st.session_state:
    st.session_state["subtitles"] = False
if "data" not in st.session_state:
    st.session_state["data"] = {}

global _  # Declare _ as global at the start of the function
_ = gettext.gettext  # Default to built-in gettext for English


# function to decorate and gather the time took by each function
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time} seconds to execute.")
        return result

    return wrapper


def validate_password():
    password_input = st.text_input(_("Enter the password please: "), type="password")
    if password_input == st.secrets["PASSWORD"]:
        st.session_state["password_ok"] = True
        return True
    elif password_input:
        st.error(_("Incorrect password. Try again."))
        return False


def check_password():
    if not st.session_state["password_ok"]:
        validate_password()
        if st.session_state["password_ok"] is True:
            st.toast(_("Correct password entered."))
            st.rerun()
    if st.session_state["password_ok"] is True:
        return True


@timer
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
        try:
            with open(mo_file_path, "rb") as mo_file:
                # Load the .mo file directly
                localizator = gettext.GNUTranslations(mo_file)
                localizator.install()
                _ = (
                    localizator.gettext
                )  # Set the _ to the gettext function from localizator
        except FileNotFoundError as e:
            st.error(f"Exception loading MO file: {e}")


def choose_language(unique_id):
    lang_choice = st.radio(
        _("Choose Language"),
        options=["English", "日本語"],
        index=0,
        key=f"lang_choice_radio_{unique_id}",  # Ensure unique key
    )

    lang_code = "en" if lang_choice == "English" else "ja"
    load_language(lang_code)

    return lang_code  # Return the selected language code

    return lang_choice  # Return the selected language choice


def display_readme(lang_code="en"):
    load_language(lang_code)  # This will set the _ function correctly
    # Use the _() function directly on the strings to be translated
    if not st.session_state["data"]:
        welcome_text = _(
            ""
            "\n"
            "# Welcome to Roland Tools\n"
            "\n"
            "This app transcribe spoken words from any language then make useful notes"
            " from it.\n"
            "    "
        )
        st.image("static/transcription.svg", width=500)
        st.markdown(welcome_text)
        file_name = "README_JP.md" if lang_code == "ja" else "README.md"
        with open(file_name, "r") as file:
            readme_content = file.read()
        with st.expander(_("I need help!")):
            st.markdown(readme_content)


def check_credentials():
    with st.sidebar:
        if "OPENAI_API_KEY" in st.secrets:
            st.toast(_("The OpenAI credentials have been entered for you!"))
            st.toast(_("You are all set!"))
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state["openai_key"] = st.secrets["OPENAI_API_KEY"]
            return True
        else:
            if not st.session_state["openai_key"]:
                openai.api_key = st.text_input(
                    _("Enter OpenAI API token:"), type="password"
                )
                if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 51):
                    st.warning(_("Please enter your credentials!"), icon="⚠️")
                    return False
                else:
                    st.session_state["openai_key"] = openai.api_key
                    st.success(_("Proceed to uploading your audio file!"), icon="👉")
                    return True


@timer
def transcription(
    file_path,
    language: str = "en",
    prompt: str = "",
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
):
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
    # return "Transcription ok"


# Function to segment the audio file
@timer
@st.cache_data
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
@timer
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
                transcription,
                file_path,
                language,
                prompt,
                response_format,
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


@timer
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
    # return "completion ok"


# Function to display full language selector and return iso code.
def get_language_choice():
    language_options = {
        "English": "en",
        "Japanese": "ja",
        "French": "fr",
        "Spanish": "es",
        "Thai": "th",
        "Arabic": "ar",
        "Chinese": "zh",
        "Korean": "ko",
    }
    selected_language_name = st.selectbox(
        _("What is the language of the audio?"),
        options=list(language_options.keys()),  # Display language names
        key="language_choice_selector",
    )
    selected_language_code = language_options[selected_language_name]
    return selected_language_code


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


@timer
def record_audio_input():
    file = {}
    recorded_file = st_audiorec()
    file["name"] = "recorded_file"
    file["file"] = recorded_file
    if recorded_file:
        return file


@timer
def upload_file():
    files = []  # list of dictionaries
    uploaded_files = st.file_uploader(
        "File Uploader",
        accept_multiple_files=True,
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
    for uf in uploaded_files:
        file_dict = {"name": uf.name, "file": uf}
        files.append(file_dict)
    return files if files else []


def upload_reader(uploads):
    if uploads:
        with st.expander("👂Audio file Uploaded"):
            for upload in uploads:
                if (
                    upload
                    and isinstance(upload, dict)
                    and "name" in upload
                    and "file" in upload
                ):
                    st.write("💽", upload["name"])
                    st.audio(upload["file"])
    else:
        st.warning("No files uploaded.")


@timer
def prepare_audio(files):
    # Check if the files list is empty or contains empty dictionaries
    if not files or all(not file_dict for file_dict in files):
        # Optionally, you can return an empty dictionary or perform some other action
        return {}

    prepared_files = {}
    for file_dict in files:
        # Check if both 'name' and 'file' keys exist in the dictionary
        if "name" in file_dict and "file" in file_dict:
            file_name = file_dict["name"]
            file_content = file_dict["file"]

            # Check if file_content is a bytes object or an UploadedFile
            if isinstance(file_content, bytes):
                # If it's already a bytes object, use it directly
                prepared_content = io.BytesIO(file_content)
            else:
                # If it's an UploadedFile, read it to get bytes
                prepared_content = io.BytesIO(file_content.read())

            prepared_files[file_name] = segment_audio(prepared_content)
        else:
            # Log or handle cases where the file_dict is not correctly formatted
            st.warning(f"Skipped an improperly formatted file entry: {file_dict}")

    return prepared_files


@timer
def transcribe_form():
    with st.form(key="transcribe_form", clear_on_submit=False, border=True):
        language = get_language_choice()
        response_format = "srt" if st.toggle(_("Transcribe to subtitles")) else "text"
        st.session_state["subtitles"] = response_format
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
            _("Transcribe Audio"),
            type="primary",
            use_container_width=True,
        )
        return transcribe_button, language, response_format, prompt


@timer
def secretary_form(prepared_prompt):
    with st.form(key="secretary_form", clear_on_submit=False, border=True):
        secretary_prompt = st.text_area(
            _("AI secretary instructions:"),
            value=(f"{prepared_prompt}"),
            help=_(
                "Describe what you want your AI secretary to do with the transcribed text: make notes, a poem, a list of groceries, etc."
            ),
        )
        model = st.radio(
            _("Model"),
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
        return process_button, is_festive, secretary_prompt, model, temperature


def set_transcription_ui():
    st.title(_("🎧🤖 Transcribe Tool"))
    tab_upload, tab_record = st.tabs([_("💽 Upload"), _("🎙️ Record")])

    files = []  # Initialize files outside the tab contexts
    with tab_upload:
        files.extend(upload_file())  # Extend the existing files list
    with tab_record:
        recorded_file = record_audio_input()
        if recorded_file:  # Check if there is a recorded file
            files.append(recorded_file)
    if files:  # Check if there are any files (uploaded or recorded)
        upload_reader(files)
        segments = prepare_audio(files)
        transcribe_button_clicked, language, response_format, prompt = transcribe_form()
        if transcribe_button_clicked:
            return segments, language, prompt, response_format
    else:
        transcribe_button_warning = st.button(
            _("Transcribe Audio"),
            type="primary",
            use_container_width=True,
        )
        if transcribe_button_warning:
            st.error(_("please upload a file to transcribe…"))


def set_secretary_ui():
    st.title(_("🤖📝 Secretary Tool"))
    prepared_prompt = get_prompt_choice() or ""
    process_button, is_festive, secretary_prompt, model, temperature = secretary_form(
        prepared_prompt
    )
    if process_button:
        return is_festive, secretary_prompt, model, temperature


def set_sidebar():
    with st.sidebar:
        col_logo, col_info = st.columns(2)
        with col_logo:
            st.image(
                "static/logo.png",
                width=200,
                use_column_width=True,
            )

        with col_info:
            # Call the function to display language selector
            language = choose_language("col_info")
        transcription_param = set_transcription_ui()
        secretary_param = set_secretary_ui()
        feedback()
        return language, transcription_param, secretary_param  # processed_text


@timer
def transcribe(files, language, prompt, response_format):
    transcribed_texts = {}
    with st.spinner(_("Wait for it... our AI is listening!")):
        st.image("static/writing.png", width=300)
        for file_name, file in files.items():
            try:
                transcribed_text = parallel_transcribe_audio(
                    file, language, prompt, response_format
                )
                transcribed_texts[file_name] = {"transcript": transcribed_text}
            except Exception as e:
                st.error(_(f"An error occurred during transcription: {e}"))
        st.session_state["data"].update(transcribed_texts)
        st.success("Done!")
        return transcribed_texts


def display_transcription(texts):
    st.title(_("🎧🤖 Transcription"))
    for index, (name, trans) in enumerate(texts.items()):
        if "transcript" in trans and trans["transcript"] is not None:
            with st.expander(f"Transcript from {name}"):
                file_extension = "srt" if st.session_state["subtitles"] else ""
                file_name = name_file(name, "transcription", format=file_extension)
                download(trans["transcript"], file_name, index)
                st.write(trans["transcript"])
                delete(name, "transcript", f"{name}transcript")


def display_notes(notes):
    st.title(_("🤖📝 Notes"))
    for index, (name, note_info) in enumerate(notes.items()):
        if (
            "note" in note_info and note_info["note"]
        ):  # Check if note exists and is not empty
            with st.expander(_(f"Secretary note from {name}")):
                file_name = name_file(name, "notes")
                download(note_info["note"], file_name, index)
                st.write(note_info["note"])
                delete(name, "note", f"{name}notes")


@timer
def secretary_process(
    transcribed_texts: dict,
    secretary_prompt: str,
    model: str,
    temperature: float,
):
    notes = transcribed_texts or {}
    with st.spinner(_("Wait for it... our AI is thinking!")):
        if transcribed_texts:
            for name, trans in transcribed_texts.items():
                completion = ""
                print("name\n", name, "\ntrans\n", trans)
                if "transcript" in trans:
                    prompt = f"```\n{trans['transcript']}``` \n{secretary_prompt}"
                else:
                    prompt = secretary_prompt
                completion = openai_completion(
                    input_text=prompt, model=model, temperature=temperature
                )
                notes[name]["note"] = completion
                print(notes)
        else:
            prompt = secretary_prompt
            completion = openai_completion(
                input_text=prompt, model=model, temperature=temperature
            )
            notes[_("AI chat")] = {}
            notes[_("AI chat")]["note"] = completion
        st.session_state["data"].update(notes)
    return notes


def name_file(file_name, *args, format="txt"):
    file_name_without_extension = file_name.rsplit(".", 1)[0]
    additional_parts = "_".join(args)  # Joining args with underscore as a separator
    if additional_parts:
        return f"{file_name_without_extension}_{additional_parts}.{format}"
    else:
        return f"{file_name_without_extension}.{format}"


@timer
def download(file, file_name, index):
    unique_key = f"download_button_{file_name}_{index}"
    st.download_button(
        label=_("Download"),
        data=file,
        file_name=file_name,
        key=unique_key,
        type="primary",
    )


def delete(file_name, key, index):
    unique_key = f"button_{file_name}_{index}"
    if st.button(label=_("Delete"), key=unique_key):
        # Check if file_name exists and key is valid
        if (
            file_name in st.session_state["data"]
            and key in st.session_state["data"][file_name]
        ):
            # Set the specific key (transcript or note) to None or empty string
            st.session_state["data"][file_name][key] = None
            # Optional: Clean up if both transcript and note are None or empty
            if not any(st.session_state["data"][file_name].values()):
                st.session_state["data"].pop(file_name)
        st.rerun()


def feedback():
    # Check if 'show_form' is already in the session state
    if "show_form" not in st.session_state:
        st.session_state["show_form"] = False

    # Button to toggle the feedback form
    if st.button(_("Write feedback")):
        st.session_state["show_form"] = not st.session_state["show_form"]

    # Feedback form will be displayed based on the 'show_form' state
    if st.session_state["show_form"]:
        with st.form(key="feedback_form"):
            body_text = ""
            sender = st.text_input(label=_("Name"), placeholder=_("Alice"))
            subject = st.text_input(
                label=_("Subject"), placeholder=_("This is awesome!")
            )
            message = "Message:\n " + st.text_area(
                label="Feedback",
                placeholder="I love everything about this app.",
                key="message",
            )
            include_data = st.checkbox(
                label="Include data",
                key="data_checkbox",
            )
            if include_data:
                body_text = (
                    message
                    + "Data:\n "
                    + (
                        str(st.session_state["data"])
                        if st.session_state["data"]
                        else "No data available"
                    )
                )
            elif not include_data:
                body_text = message + "\n\nNo data shared"
            submit_feedback = st.form_submit_button(_("Send Feedback"))
            if submit_feedback:
                if (
                    sender and subject and body_text
                ):  # Simple validation to ensure fields are filled
                    with st.spinner(_("Sending feedback...")):
                        send_email(sender, subject, body_text)
                        st.success(_("Thank you for your feedback!"))
                else:
                    st.error(_("Please fill in all fields."))


def send_email(sender, subject, body_text):
    SMTP_SERVER = st.secrets["SMTP_SERVER"]
    SMTP_PORT = st.secrets["SMTP_PORT"]
    EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
    print("Parameters", SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD)
    print("arguments", sender, subject, body_text)
    message = MIMEMultipart()
    message["From"] = st.secrets["EMAIL_ADDRESS"]
    message["To"] = st.secrets["EMAIL_ADDRESS"]
    message["Subject"] = f"Roland's Tool feedback from {sender} on {subject}"
    message.attach(MIMEText(body_text, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(message)

    print("Email sent successfully")


def main():
    if not check_password():
        st.stop()
    # choose_language()
    if not st.session_state["openai_key"] and not check_credentials():
        st.stop()

    language, transcription_param, secretary_param = set_sidebar()

    # Get the selected language from the sidebar and then display the README in the main section
    display_readme(language)
    transcribed_texts = {}
    notes = {}

    # Transcription
    if transcription_param:
        files, language, response_format, prompt = transcription_param
        transcribed_texts = transcribe(files, language, response_format, prompt)
        st.session_state["data"].update(transcribed_texts)

    # Secretary Processing
    if secretary_param:
        is_festive, secretary_prompt, model, temperature = secretary_param
        notes = secretary_process(
            st.session_state["data"] if st.session_state["data"] else {},
            secretary_prompt,
            model,
            temperature,
        )
        st.session_state["data"].update(notes)
        if is_festive:
            st.balloons()

    # Display Logic
    if st.session_state["data"]:
        data_has_transcript = any(
            "transcript" in item for item in st.session_state["data"].values()
        )
        data_has_note = any(
            "note" in item for item in st.session_state["data"].values()
        )

        if data_has_transcript and data_has_note:
            col_trans, col_note = st.columns(2)
            with col_trans:
                display_transcription(st.session_state["data"])
            with col_note:
                display_notes(st.session_state["data"])
        elif data_has_transcript:
            display_transcription(st.session_state["data"])
        elif data_has_note:
            display_notes(st.session_state["data"])
            st.image("static/thumbsup.png", width=300)


if __name__ == "__main__":
    main()
