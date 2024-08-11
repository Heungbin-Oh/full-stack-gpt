import streamlit as st
from utils.authentication import check_login
from utils.utils import make_dir

# Set up the page configuration
st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
)

# To store cache data, make sure the folder exist
make_dir()
# Check if the user is logged in
check_login()

username = st.session_state['username']
login_result = f"Welcome {username}!"
# logout button
st.success(login_result)
with st.sidebar:
    st.info(login_result)
    logout = st.button('Logout')
    if logout:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()

st.markdown(
    """
    # Hello!

    Welcome to my FullstackGPT Portfolio!

    Here are the apps I made:

    - [ ] [DocumentGPT](/DocumentGPT)
    - [ ] [PrivateGPT](/PrivateGPT)
    - [ ] [QuizGPT](/QuizGPT)
    - [ ] [SiteGPT](/SiteGPT)
    - [ ] [MeetingGPT](/MeetingGPT)
    - [ ] [InvestorGPT](/InvestorGPT)
    """
)
