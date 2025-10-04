import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv(
    "DB_PATH",
    r"C:\\Users\\abdul\\OneDrive\\Documents\\GitHub\\nasa-dashboard-backend\\biodash.db",
)
SECTION_POLICY = tuple(
    s.strip() for s in os.getenv("SECTION_POLICY", "abstract,results,discussion").split(",") if s.strip()
)
RESULTS_BOOST = float(os.getenv("RESULTS_BOOST", 1.3))
DISCUSSION_BOOST = float(os.getenv("DISCUSSION_BOOST", 1.2))
MIN_CHARS = int(os.getenv("MIN_CHARS", 300))
