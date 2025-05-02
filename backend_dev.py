from flask import Flask, request
import sqlite3
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
import random
import string
import os

app = Flask(__name__)
app.config['PREFERRED_URL_SCHEME'] = 'https'
CORS(app)

# Simple in-memory store for recovery codes (for demo; use persistent store in production)
recovery_codes = {}


def get_db_connection():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn


# Create users table if it doesn't exist
def init_db():
    conn = get_db_connection()
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        config_json TEXT
    )""")
    conn.commit()
    conn.close()


init_db()


# Helper to send email (configure SMTP as needed)
def send_email(to_email, subject, body):
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    if not smtp_user or not smtp_pass:
        print("SMTP credentials not set.")
        return False
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(smtp_user, to_email, message)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


@app.route("/register", methods=["POST"])
def register():
    data = request.form
    username = data.get("username")
    password = data.get("password")
    email = data.get("email")
    if not username or not password or not email:
        return "Username, password, and email required.", 400
    conn = get_db_connection()
    cur = conn.cursor()
    # Check if username or email already exists
    cur.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    if cur.fetchone():
        conn.close()
        return "Username or email is already taken.", 409
    hashed_pw = generate_password_hash(password)
    # Check if password hash already exists
    cur.execute("SELECT * FROM users WHERE password = ?", (hashed_pw,))
    if cur.fetchone():
        conn.close()
        return "Password is already used by another user. Please choose a different password.", 409
    cur.execute(
        "INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hashed_pw, email)
    )
    conn.commit()
    conn.close()
    return "Registered successfully.", 201


@app.route("/login", methods=["POST"])
def login():
    data = request.form
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return "Username and password required.", 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    if user and check_password_hash(user["password"], password):
        return "Logged in successfully.", 200
    return "Invalid username or password.", 401


@app.route("/recover_request", methods=["POST"])
def recover_request():
    data = request.form
    email = data.get("email")
    if not email:
        return "Email required.", 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    if not user:
        return "Email not found.", 404
    code = ''.join(random.choices(string.digits, k=6))
    recovery_codes[email] = code
    send_email(email, "Password Recovery Code", f"Your recovery code is: {code}")
    return "Recovery code sent to your email.", 200


@app.route("/recover_confirm", methods=["POST"])
def recover_confirm():
    data = request.form
    email = data.get("email")
    code = data.get("code")
    new_password = data.get("new_password")
    if not email or not code or not new_password:
        return "Email, code, and new password required.", 400
    if recovery_codes.get(email) != code:
        return "Invalid or expired code.", 400
    hashed_pw = generate_password_hash(new_password)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
    conn.commit()
    conn.close()
    del recovery_codes[email]
    return "Password updated successfully.", 200


@app.route("/save_config", methods=["POST"])
def save_config():
    data = request.json
    username = data.get("username")
    config_json = data.get("config_json")
    if not username or config_json is None:
        return "Username and config_json required.", 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET config_json = ? WHERE username = ?", (config_json, username)
    )
    conn.commit()
    conn.close()
    return "Config saved.", 200


@app.route("/get_config", methods=["GET"])
def get_config():
    username = request.args.get("username")
    if not username:
        return "Username required.", 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT config_json FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row["config_json"]:
        return row["config_json"], 200
    else:
        return "{}", 200


@app.errorhandler(404)
def page_not_found(e):
    return {"error": "Page not found"}, 404


@app.errorhandler(500)
def internal_server_error(e):
    return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    app.run(port=5000,
            debug=True,
            use_reloader=True,
            threaded=True)
