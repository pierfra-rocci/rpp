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
    """Establish and return a connection to the SQLite database.

    Returns:
        sqlite3.Connection: A database connection object with row_factory set
                            to sqlite3.Row for dictionary-like row access.
    """
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database by creating the 'users' table if it doesn't exist.

    The table stores user information including username, hashed password,
    email, and a JSON string for user-specific configuration.
    """
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


def send_email(to_email, subject, body):
    """Send an email using SMTP configuration from environment variables.

    Reads SMTP server, port, user, and password from environment variables
    (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS). Uses TLS for security.

    Args:
        to_email (str): The recipient's email address.
        subject (str): The subject line of the email.
        body (str): The plain text body of the email.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
              Prints error messages to the console on failure or if SMTP
              credentials are not set.
    """
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
    """Handle user registration requests.

    Expects 'username', 'password', and 'email' in the form data.
    Hashes the password before storing it. Checks for existing username/email
    and prevents duplicate password hashes.

    Returns:
        tuple: (message, status_code)
               - ("Registered successfully.", 201) on success.
               - ("Username, password, and email required.", 400) if data missing.
               - ("Username or email is already taken.", 409) if conflict.
               - ("Password is already used...", 409) if hash conflict.
    """
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
    """Handle user login requests.

    Expects 'username' and 'password' in the form data.
    Compares the provided password against the stored hash.

    Returns:
        tuple: (message, status_code)
               - ("Logged in successfully.", 200) on success.
               - ("Username and password required.", 400) if data missing.
               - ("Invalid username or password.", 401) on failure.
    """
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
    """Handle the first step of password recovery: requesting a code.

    Expects 'email' in the form data.
    Generates a 6-digit recovery code, stores it temporarily (in-memory),
    and sends it to the user's email.

    Returns:
        tuple: (message, status_code)
               - ("Recovery code sent to your email.", 200) on success.
               - ("Email required.", 400) if email missing.
               - ("Email not found.", 404) if email not in database.
    """
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
    """Handle the second step of password recovery: confirming the code and setting a new password.

    Expects 'email', 'code', and 'new_password' in the form data.
    Verifies the recovery code and updates the user's password hash in the database.
    Removes the used recovery code.

    Returns:
        tuple: (message, status_code)
               - ("Password updated successfully.", 200) on success.
               - ("Email, code, and new password required.", 400) if data missing.
               - ("Invalid or expired code.", 400) if code is incorrect.
    """
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
    """Save user-specific configuration (as JSON string) to the database.

    Expects JSON payload with 'username' and 'config_json'.

    Returns:
        tuple: (message, status_code)
               - ("Config saved.", 200) on success.
               - ("Username and config_json required.", 400) if data missing.
    """
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
    """Retrieve user-specific configuration (as JSON string) from the database.

    Expects 'username' as a query parameter.

    Returns:
        tuple: (json_string or empty_json, status_code)
               - (config_json, 200) if config found.
               - ("{}", 200) if no config found for the user.
               - ("Username required.", 400) if username missing.
    """
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
    """Custom error handler for 404 Not Found errors.

    Args:
        e: The error object.

    Returns:
        tuple: (json_response, status_code)
               - ({"error": "Page not found"}, 404)
    """
    return {"error": "Page not found"}, 404


@app.errorhandler(500)
def internal_server_error(e):
    """Custom error handler for 500 Internal Server errors.

    Args:
        e: The error object.

    Returns:
        tuple: (json_response, status_code)
               - ({"error": "Internal server error"}, 500)
    """
    return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    app.run(port=5000,
            debug=True,
            use_reloader=True,
            threaded=True)
