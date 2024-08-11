import hashlib
import sqlite3
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the admin key from environment variables
ADMIN_KEY = os.getenv("ADMIN_KEY")
# Initialize SQLite Database


def init_db():
    connect = sqlite3.connect('users.db')
    c = connect.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    connect.commit()
    connect.close()

# Hashing password for security


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Add a new user to the database


def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
              (username, hash_password(password)))
    conn.commit()
    conn.close()

# Validate if user exists in the database


def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?',
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user


def login():
    st.subheader("Login")
    with st.form(key='login_form'):
        username = st.text_input(
            "User Name", key="login_username", placeholder="Enter Username")
        password = st.text_input(
            "Password", type='password', key="login_password", placeholder="Enter Password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            user = login_user(username, password)
            if user:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Logged In as {username}")
                st.rerun()
            else:
                st.warning("Incorrect Username/Password")

    if st.button('Sign Up', key="signup_button"):
        st.session_state['signup_mode'] = True
        st.rerun()


def signup():
    st.subheader("Create New Account")

    with st.form(key='signup_form'):
        admin_key = st.text_input(
            "Admin Key", type='password', key="admin_key_input", placeholder="Enter Admin key...      Hint: My Birthday!")
        new_user = st.text_input(
            "Username", key="signup_username", placeholder="Enter username")
        new_password = st.text_input(
            "Password", type='password', key="signup_password", placeholder="Enter password")
        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            if admin_key == ADMIN_KEY:
                try:
                    register_user(new_user, new_password)
                    st.success("You have successfully created an account")
                    st.info("Returning to Login Menu")
                    st.session_state['signup_mode'] = False
                    st.rerun()  # Return to the login page after successful sign-up
                except sqlite3.IntegrityError:
                    st.warning("Username already exists")


def main_page():
    init_db()

    if 'signup_mode' not in st.session_state:
        st.session_state['signup_mode'] = False

    if st.session_state['signup_mode']:
        signup()
    else:
        login()


def check_login():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("Please log in to access this page.")
        main_page()
        st.stop()
