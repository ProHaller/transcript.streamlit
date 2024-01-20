import streamlit as st
import smtplib
import databases
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from streamlit_option_menu import option_menu
from streamlit.runtime.state import session_state
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


if "authenticator" not in st.session_state:
    st.session_state["authenticator"] = None
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

global authenticator


def initiate():
    with open("./config.yaml", "r") as file:
        config = yaml.load(file, Loader=SafeLoader)
    st.session_state["config"] = config

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config["preauthorized"],
    )
    st.session_state["authenticator"] = authenticator
    return authenticator


def login():
    st.image("static/sorry.png", width=300)
    st.warning(
        "I apologize for the login troubles, You can reset your password with the fogot password function."
    )
    st.session_state["authenticator"].login("Login", "main")
    # Handling different authentication states
    if st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")


def register():
    try:
        if st.session_state["authenticator"].register_user(
            "Register user", preauthorization=False
        ):
            # Access config from session state
            config = st.session_state["config"]
            with open("./config.yaml", "w") as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success("User registered successfully")
    except Exception as e:
        st.error(e)


def forgot_password():
    try:
        (
            username_of_forgotten_password,
            email_of_forgotten_password,
            new_random_password,
        ) = st.session_state["authenticator"].forgot_password("Forgot password")
        if username_of_forgotten_password:
            # Random password should be transferred to user securely
            body_text = f"Hello {username_of_forgotten_password},\nYour new password is {new_random_password}\nEnjoy"
            send_email(
                "password",
                body_text,
                email_of_forgotten_password,
            )
            st.success("New password has been sent to your email address.")
        else:
            st.error("Username not found")
    except Exception as e:
        st.error(e)


def forgot_username():
    try:
        (
            username_of_forgotten_username,
            email_of_forgotten_username,
        ) = st.session_state["authenticator"].forgot_username("Forgot username")
        if username_of_forgotten_username:
            # Username should be transferred to user securely
            body_text = (
                f"Hello, \nYour username is {username_of_forgotten_username}\nEnjoy"
            )
            send_email(
                "username",
                body_text,
                email_of_forgotten_username,
            )
            st.success("Your Username has been sent to your email address.")
        else:
            st.error("Email not found")
    except Exception as e:
        st.warning(e)


def change_password():
    if st.session_state["authentication_status"]:
        try:
            if st.session_state["authenticator"].reset_password(
                st.session_state["username"], "Change password"
            ):
                st.success("Password modified successfully")
        except Exception as e:
            st.error(e)


def logout():
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.session_state["authenticator"].logout("Logout", "main", key="unique_key")


def send_email(subject, body_text, email_to):
    SMTP_SERVER = st.secrets["SMTP_SERVER"]
    SMTP_PORT = st.secrets["SMTP_PORT"]
    EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
    print("Parameters", SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD)
    message = MIMEMultipart()
    message["From"] = st.secrets["EMAIL_ADDRESS"]
    message["To"] = email_to
    message["Subject"] = f"Roland's Tool {subject} reminder"
    message.attach(MIMEText(body_text, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(message)

    print("Email sent successfully")


def main():
    if (
        "authenticator" not in st.session_state
        or st.session_state["authenticator"] is None
    ):
        st.session_state["authenticator"] = initiate()
    if st.session_state["authentication_status"] is None:
        menu = option_menu(
            None,
            ["Login", "Register", "Forgot something?"],
            icons=["door-open", "cloud-upload", "search"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        if menu == "Register":
            register()
        elif menu == "Forgot something?":
            col_pass, col_user = st.columns(2)
            with col_pass:
                forgot_password()
            with col_user:
                forgot_username()
        elif menu == "Login":
            login()
    databases.main()
    if st.session_state["authentication_status"]:
        logged_in = st.session_state["authentication_status"]
        return logged_in, st.session_state["authenticator"]
    else:
        return False


if __name__ == "__main__":
    main()
