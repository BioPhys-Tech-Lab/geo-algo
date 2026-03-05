import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run_server():
    # Ensure we are in the root directory of the project
    # (where dashboard/ and output/ are siblings)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Rock Research Dashboard active at: http://localhost:{PORT}/dashboard/")
        print("Press Ctrl+C to stop the server.")
        webbrowser.open(f"http://localhost:{PORT}/dashboard/")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
