# DocumentGPT

A Streamlit application to interact with documents using OpenAI's GPT-3.5 and LangChain.

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

3. **Create a `.streamlit/config.toml` file** to configure Streamlit settings (optional):
    ```toml
    [server]
    headless = true
    ```

4. **Run the Streamlit app**:
    ```bash
    streamlit run Home.py
    ```

5. **API Key Management**:
    - If you have the API keys saved in `~/.api_keys.json`, they will be loaded automatically.
    - If the keys are not found, you will be prompted to enter your API keys in the sidebar. These keys will be saved for future use.

## API Key File Format

The API keys are saved in a JSON file located at `~/.api_keys.json`. The format of this file is as follows:

```json
{
    "OPENAI_API_KEY": "your_openai_api_key",
    "LANGCHAIN_API_KEY": "your_langchain_api_key"
}
