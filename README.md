streamlit run app.py

# Book Summarization Application

This is a web application for generating concise summaries of long texts and documents. The system leverages large language models (LLMs) through the LangChain and Hugging Face Transformers libraries, providing a simple interface for end-users.

## Features

  - Summarizes long-form text documents.
  - Handles documents exceeding model context limits via automated text chunking.
  - Simple and interactive web interface for ease of use.
  - Built on a modular and scalable Python architecture.

## Technology Stack

  - **Backend:** Python
  - **Web Framework:** Streamlit
  - **LLM Orchestration:** LangChain
  - **NLP Models:** Hugging Face Transformers
  - **Text Processing:** spaCy

## Project Structure

```
summarization_project/
├── app.py              # Streamlit front-end application
├── summarizer.py       # Core summarization logic and NLP pipeline
├── requirements.txt    # Project dependencies
└── venv/               # Virtual environment directory (optional)
```

## Setup and Installation

Follow these steps to set up the project locally.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/summarization_project.git
    cd summarization_project
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the necessary spaCy model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

Once the setup is complete, run the following command from the root of the project directory to start the application:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## License

This project is licensed under the MIT License.