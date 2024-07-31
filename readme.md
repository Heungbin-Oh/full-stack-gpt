
Python

Framework: LangChain

# DocumentGPT

A Streamlit application to interact with documents using OpenAI's GPT-3.

## Setup

1. Clone the repository.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.streamlit/secrets.toml` file with your API keys:

    ```toml
    [credentials]
    OPENAI_API_KEY = "your_openai_api_key"
    LANGCHAIN_API_KEY = "your_langchain_api_key"
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Deployment

Follow the instructions to deploy on Streamlit Cloud.
