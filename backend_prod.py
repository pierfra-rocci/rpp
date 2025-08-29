from waitress import serve
from backend import app

if __name__ == "__main__":
    serve(app, port=5000, threads=8, connection_limit=100)
