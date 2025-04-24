from flask import Flask, request
import sqlite3
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create users table if it doesn't exist
def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    # Add config_json column if not exists
    try:
        conn.execute('ALTER TABLE users ADD COLUMN config_json TEXT')
    except Exception:
        pass  # Already exists
    conn.commit()
    conn.close()

init_db()

@app.route('/register', methods=['POST'])
def register():
    data = request.form
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return 'Username and password required.', 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = ?', (username,))
    if cur.fetchone():
        conn.close()
        return 'Username is already taken.', 409
    cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()
    return 'Registered successfully.', 201

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return 'Username and password required.', 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    if cur.fetchone():
        conn.close()
        return 'Logged in successfully.', 200
    conn.close()
    return 'Invalid username or password.', 401

@app.route('/recover', methods=['POST'])
def recover():
    data = request.form
    username = data.get('username')
    new_password = data.get('new_password')
    if not username or not new_password:
        return 'Username and new password required.', 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = ?', (username,))
    if not cur.fetchone():
        conn.close()
        return 'Username does not exist.', 404
    cur.execute('UPDATE users SET password = ? WHERE username = ?', (new_password, username))
    conn.commit()
    conn.close()
    return 'Password updated successfully.', 200

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.json
    username = data.get('username')
    config_json = data.get('config_json')
    if not username or config_json is None:
        return 'Username and config_json required.', 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('UPDATE users SET config_json = ? WHERE username = ?', (config_json, username))
    conn.commit()
    conn.close()
    return 'Config saved.', 200

@app.route('/get_config', methods=['GET'])
def get_config():
    username = request.args.get('username')
    if not username:
        return 'Username required.', 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT config_json FROM users WHERE username = ?', (username,))
    row = cur.fetchone()
    conn.close()
    if row and row['config_json']:
        return row['config_json'], 200
    else:
        return '{}', 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)
