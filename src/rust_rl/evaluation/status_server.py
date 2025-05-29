"""
Status Server

A simple HTTP server that provides detailed status information about the vLLM server
including model loading state, even when the main vLLM server isn't ready yet.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .config import UnifiedConfig
from .dynamic_model_server import DynamicModelServer


class StatusHandler(BaseHTTPRequestHandler):
    """HTTP handler for status requests"""
    
    dynamic_server = None  # Class variable to hold the dynamic server instance
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/status":
            self.send_status()
        elif self.path == "/health":
            self.send_health()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def send_status(self):
        """Send detailed status information"""
        status = self.dynamic_server.get_status()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode())
    
    def send_health(self):
        """Send simple health check"""
        status = self.dynamic_server.get_status()
        if status["running"]:
            self.send_response(200)
            response = {"status": "healthy", "model": status["current_model"]}
        elif status["process_alive"]:
            self.send_response(202)  # Accepted, but not ready
            response = {"status": "loading", "model": status["current_model"]}
        else:
            self.send_response(503)  # Service unavailable
            response = {"status": "stopped", "model": None}
        
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


class StatusServer:
    """Status server that runs alongside vLLM server"""
    
    def __init__(self, dynamic_server: DynamicModelServer, port: int = 8000):
        self.dynamic_server = dynamic_server
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the status server"""
        # Set the dynamic server as a class variable
        StatusHandler.dynamic_server = self.dynamic_server
        
        self.server = HTTPServer(("localhost", self.port), StatusHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"📊 Status server started at http://localhost:{self.port}")
        print(f"   Status endpoint: http://localhost:{self.port}/status")
        print(f"   Health endpoint: http://localhost:{self.port}/health")
    
    def stop(self):
        """Stop the status server"""
        if self.server:
            self.server.shutdown()
            self.server = None
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
