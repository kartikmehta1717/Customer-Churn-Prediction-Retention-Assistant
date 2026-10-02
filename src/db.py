import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_engine():
    user = os.getenv("DB_USER", "root")
    pwd = quote_plus(os.getenv("DB_PASSWORD", ""))
    host = os.getenv("DB_HOST", "localhost")
    name = os.getenv("DB_NAME", "churn_db")
    return create_engine(f"mysql+mysqlconnector://{user}:{pwd}@{host}/{name}")