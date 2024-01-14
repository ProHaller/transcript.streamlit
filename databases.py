from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm.exc import NoResultFound
import streamlit as st
from datetime import datetime

# from datetime import datetime

# Database connection details
DB_URL = st.secrets["DATABASE_URL"]

# Setting up SQLAlchemy
engine = create_engine(DB_URL)
Session = sessionmaker(engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now())
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False, default="Unknown")


class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    file_name = Column(String(255), nullable=False)
    user = relationship("User")


class Transcript(Base):
    __tablename__ = "transcripts"
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("uploaded_files.id"))
    transcript_text = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now())
    file = relationship("UploadedFile")


class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("uploaded_files.id"))
    note_text = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now())
    file = relationship("UploadedFile")


# Function to get user info
def get_user_info(username):
    with Session() as session:
        return session.query(User).filter(User.username == username).first()


def get_user_data(username):
    with Session() as session:
        # Fetch the user based on the username
        user = session.query(User).filter(User.username == username).first()

        if not user:
            return None  # User not found

        user_info = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "name": user.name,
        }

        files_data = []

        # Fetch all files uploaded by the user
        files = (
            session.query(UploadedFile).filter(UploadedFile.user_id == user.id).all()
        )

        for file in files:
            file_info = {
                "file_id": file.id,
                "file_name": file.file_name,
                "created_at": file.created_at,
                "transcripts": [],
                "notes": [],
            }

            # Fetch transcripts for each file
            transcripts = (
                session.query(Transcript).filter(Transcript.file_id == file.id).all()
            )
            file_info["transcripts"] = [
                {
                    "id": transcript.id,
                    "transcript_text": transcript.transcript_text,
                    "created_at": transcript.created_at,
                    "updated_at": transcript.updated_at,
                }
                for transcript in transcripts
            ]

            # Fetch notes for each file
            notes = session.query(Note).filter(Note.file_id == file.id).all()
            file_info["notes"] = [
                {
                    "id": note.id,
                    "note_text": note.note_text,
                    "created_at": note.created_at,
                    "updated_at": note.updated_at,
                }
                for note in notes
            ]

            files_data.append(file_info)

        user_data = {"user_info": user_info, "files": files_data}

        return user_data


# Function to list user files
def list_user_files(user_id):
    with Session() as session:
        return session.query(UploadedFile).filter(UploadedFile.user_id == user_id).all()


# Function to get transcripts for a file
def get_transcripts(file_id):
    with Session() as session:
        return session.query(Transcript).filter(Transcript.file_id == file_id).all()


# Function to get notes for a file
def get_notes(file_id):
    with Session() as session:
        return session.query(Note).filter(Note.file_id == file_id).all()


# Function to upload a file if it doesn't already exist
def add_uploaded_file(user_id, file_name):
    with Session() as session:
        # Check if a file with the same user_id and file_name already exists
        try:
            existing_file = (
                session.query(UploadedFile)
                .filter_by(user_id=user_id, file_name=file_name)
                .one()
            )
            return existing_file.id
        except NoResultFound:
            # If no such file exists, create a new one
            new_file = UploadedFile(user_id=user_id, file_name=file_name)
            session.add(new_file)
            session.commit()
            return new_file.id


# Function to upsert a transcript
def upsert_transcript(file_id, transcript_text):
    with Session() as session:
        # Check if a transcript with the given file_id exists
        transcript = (
            session.query(Transcript).filter(Transcript.file_id == file_id).first()
        )
        if transcript:
            # If exists, update the existing transcript
            transcript.transcript_text = transcript_text
        else:
            # If not, create a new transcript
            new_transcript = Transcript(
                file_id=file_id, transcript_text=transcript_text
            )
            session.add(new_transcript)
        session.commit()
        return transcript.id if transcript else new_transcript.id


# Function to upsert a note
def upsert_note(file_id, note_text):
    with Session() as session:
        note = session.query(Note).filter(Note.file_id == file_id).first()

        if note:
            note.note_text = note_text
        else:
            new_note = Note(file_id=file_id, note_text=note_text)
            session.add(new_note)

        session.commit()
        return note.id if note else new_note.id


def upsert_user(username, email, name):
    with Session() as session:
        # Check if the user exists and if the data is the same
        existing_user = session.query(User).filter(User.username == username).first()

        if existing_user:
            # Update the user if the email or name is different
            if existing_user.email != email or existing_user.name != name:
                existing_user.email = email
                existing_user.name = name
                session.commit()
            return existing_user
        else:
            # Insert new user as it does not exist
            new_user = User(username=username, email=email, name=name)
            session.add(new_user)
            session.commit()
            return new_user


# Function to update user information
def update_user_info(user_id, email, name):
    with Session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if user:
            user.email = email
            user.name = name
            session.commit()


def set_user():
    if st.session_state["authentication_status"] is True:
        credentials = st.session_state["authenticator"].credentials
        user_info = upsert_user(
            st.session_state["authenticator"].username,
            credentials["usernames"][st.session_state["authenticator"].username][
                "email"
            ],
            credentials["usernames"][st.session_state["authenticator"].username][
                "name"
            ],
        )
        st.session_state["user"] = user_info


def main():
    set_user()
    # get user Data
    # todo Clean the data and display in on demand.


if __name__ == "__main__":
    main()
