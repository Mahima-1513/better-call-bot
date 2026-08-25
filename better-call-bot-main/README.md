# Better Call Bot

**Better Call Bot** is an Indian legal-information chatbot. It helps users explore general legal information from a local document knowledge base in a simple chat interface.

> This project provides general information only. It is not a substitute for advice from a qualified lawyer.

## What it can do

- Answer general questions related to Indian legal information
- Search a FAISS vector database built from legal documents
- Keep the conversation focused and easy to understand
- Show clearly separated user queries and assistant responses

## Run locally

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file and add your Groq API key:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Tech stack

Streamlit · LangChain · Groq · FAISS · Hugging Face embeddings

## Deployment

The app can be deployed on Streamlit Community Cloud. Add `GROQ_API_KEY` in the app's **Secrets** settings; never upload your `.env` file or API key to GitHub.
