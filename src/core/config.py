from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Singleton LLM instance to be shared across graphs
# configured with robust retry logic for "RESOURCE_EXHAUSTED" (429) errors.
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    max_retries=10, # Aggressive retries for free tier/preview limits
    request_timeout=60,
)
