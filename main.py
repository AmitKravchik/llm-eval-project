from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

res = llm.invoke("Each one of my 3 friends brought me 5 oranges. How many oranges do I have?")
print(res.content)
