"""User authentication: signup, login, DuckDB-backed."""
import re
import bcrypt
from datetime import datetime
from src.db import get_connection

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def init_users_table() -> None:
    con = get_connection()
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY,
            email         VARCHAR UNIQUE NOT NULL,
            password_hash VARCHAR NOT NULL,
            created_at    TIMESTAMP NOT NULL
        )
    """)
    con.execute("CREATE SEQUENCE IF NOT EXISTS users_id_seq START 1")
    con.close()


def _hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), pw_hash.encode())
    except ValueError:
        return False


def signup(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return False, "Invalid email format."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    con = get_connection()
    if con.execute("SELECT 1 FROM users WHERE email = ?", [email]).fetchone():
        con.close()
        return False, "Email already registered."

    con.execute(
        "INSERT INTO users (id, email, password_hash, created_at) "
        "VALUES (nextval('users_id_seq'), ?, ?, ?)",
        [email, _hash_password(password), datetime.utcnow()],
    )
    con.close()
    return True, "Account created. You can now log in."


def login(email: str, password: str) -> tuple[bool, str]:
    email = email.strip().lower()
    con = get_connection()
    row = con.execute("SELECT password_hash FROM users WHERE email = ?", [email]).fetchone()
    con.close()
    if row is None:
        return False, "Email not found."
    if not _verify_password(password, row[0]):
        return False, "Incorrect password."
    return True, "Logged in."