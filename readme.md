# Fullstack GPT

A Streamlit application to interact with documents, audio files, etc using OpenAI's GPT-4o mini and LangChain.

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/document-gpt.git
   cd document-gpt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Create a `.streamlit/secrets.toml` file** to save API keys; OPENAI_API_KEY, LANGCHAIN_API_KEY:
   ```toml
   OPENAI_API_KEY = "your_api_key"
   LANGCHAIN_TRACING_V2 = true
   LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
   LANGCHAIN_API_KEY = "your_api_key"
   LANGCHAIN_PROJECT = "Fullstack-GPT"
   ADMIN_KEY="you_can_set_your_admin_key_to_make_new_account"
   ```
4. **Create a `.streamlit/config.toml` file** to configure Streamlit settings (optional):

   ```toml
   [server]
   headless = true
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run Home.py
   ```
