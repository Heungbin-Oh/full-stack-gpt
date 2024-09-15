# Fullstack GPT
![GPT_image](https://github.com/user-attachments/assets/b91d27a4-1983-4de1-90e0-fc2e7f7f4da8)

Created 6 Fullstack GPT Web Services using OpenAI's GPT-4o mini, LangChain, and Streamlit; DocumentGPT, QuizGPT, PrivateGPT, SiteGPT, MeetingGPT, and InvestorGPT.

**#### !admin code to make an account!; 0622**

###### Deployment Link: https://full-stack-gpt-heungbin-oh.streamlit.app/


## What Used in the Project

- **Python & OpenAI API**: Selected to practice and understand about Langchain and Lage Language Model model.
- **Streamlit Framework**: Selected to build UI with Python code and to deploy the app to the Streamlit Cloud.

## Features

- Used 3 different chain methods; Stuff, Re-Rank, Refine chain.
- Utilized an Agent from Langchain to search for the company information.
- Used Embeddings and Vector store to reduce memory usage.

## Challenges & Future Improvements

### Challenges:

Had difficulty with User Authentication. I can simply follow the other things with the documentation of Langchain. So, there was not so much difficulty. On the Other hand, for the user authentication, I have never created this thing. Therefore, I used a simple database with SQLite and made it a util function.

### Future Improvements:

Implement a new service with ChatGPT Plugin using FastAPI. By using the plugin, I can use authentication service from it and reduce the length of my code.
