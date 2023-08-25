import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve the values
OUTPUT_DIR = os.getenv("OUTPUT_DIR", default=os.getcwd())
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DATABASE_ID = os.getenv("DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
