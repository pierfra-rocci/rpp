from waitress import serve
from backend import app

if __name__ == "__main__":
    serve(app, port=5000,
          threads=True, connection_limit=100)
