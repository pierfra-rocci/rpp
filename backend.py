from flask import Flask, request
import sqlite3
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
import random
import string
import os
import base64
import datetime

app = Flask(__name__)
app.config["PREFERRED_URL_SCHEME"] = "https"
CORS(app)

# Use absolute path for database file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Create users table if it doesn't exist
def init_db():
    try:
        conn = get_db_connection()
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            config_json TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS recovery_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            expires_at DATETIME NOT NULL
        )""")
        conn.commit()
        print("Database schema created successfully")

        # Verify the tables exist
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        if cursor.fetchone():
            print("Confirmed 'users' table exists")
        else:
            print("WARNING: 'users' table was not created!")
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recovery_codes'"
        )
        if cursor.fetchone():
            print("Confirmed 'recovery_codes' table exists")
        else:
            print("WARNING: 'recovery_codes' table was not created!")

        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise


init_db()


# Helper send email (configure SMTP as needed)
def send_email(to_email, subject, body):
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass_encoded = os.environ.get("SMTP_PASS_ENCODED")
    smtp_pass = (
        base64.b64decode(smtp_pass_encoded).decode() if smtp_pass_encoded else None
    )
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
    except smtplib.SMTPAuthenticationError as e:
        print(f"Failed to send email: SMTP authentication error - {e}")
        return False
    except smtplib.SMTPConnectError as e:
        print(f"Failed to send email: Error connecting to the SMTP server - {e}")
        return False
    except Exception as e:
        print(f"Failed to send email: An unexpected error occurred - {e}")
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
    cur.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?", (username, email)
    )
    if cur.fetchone():
        conn.close()
        return "Username or email is already taken.", 409
    hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
    # Check if password hash already exists
    cur.execute("SELECT * FROM users WHERE password = ?", (hashed_pw,))
    if cur.fetchone():
        conn.close()
        return (
            "Password is already used by another user. Please choose a different password.",
            409,
        )
    cur.execute(
        "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
        (username, hashed_pw, email),
    )
    conn.commit()
    conn.close()
    # Send welcome email
    send_email(
        email,
        "Welcome to RAPAS Photometry Pipeline!",
        f"Hi {username},\n\nThank you for registering for the RAPAS Photometry Pipeline. We are excited to have you on board!\n\nBest,\nThe RAPAS Team",
    )
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
    if not user:
        conn.close()
        return "Email not found.", 404
    code = "".join(random.choices(string.digits, k=6))
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=15)
    hashed_code = generate_password_hash(code)
    cur.execute(
        "INSERT INTO recovery_codes (email, code, expires_at) VALUES (?, ?, ?)",
        (email, hashed_code, expires_at),
    )
    conn.commit()
    conn.close()
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
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM recovery_codes WHERE email = ?", (email,))
    recovery_codes = cur.fetchall()
    
    valid_code_found = False
    for recovery_code in recovery_codes:
        if check_password_hash(recovery_code["code"], code):
            expires_at = datetime.datetime.strptime(recovery_code["expires_at"], "%Y-%m-%d %H:%M:%S.%f")
            if datetime.datetime.now() > expires_at:
                continue  # Expired code
            valid_code_found = True
            break

    if not valid_code_found:
        conn.close()
        return "Invalid or expired recovery code.", 400

    hashed_pw = generate_password_hash(new_password, method='pbkdf2:sha256')
    cur.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, email))
    cur.execute("DELETE FROM recovery_codes WHERE email = ?", (email,))
    conn.commit()
    conn.close()
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
    app.run(port=5000, debug=True, use_reloader=True, threaded=True)
