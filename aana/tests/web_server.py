# """Programmatically configured HTTP server for running tests"""
# import random
# import socket
# import threading
# from http.server import BaseHTTPRequestHandler, HTTPServer


# class TestWebServerStd:
#     def __init__(self, name: str = 'test', *, port: str = '', host: str = ''):
#         self.endpoints = {}
#         self.host = host or self.get_host()
#         self.port = port or str(self.find_port())
#         self.web_server = None

#     def add_endpoint(self, endpoint: str, λ: 'Callable', content_type: str):
#         if endpoint in self.endpoints:
#             raise RuntimeError('Endpoint %s already defined' % endpoint)
#         self.endpoints[endpoint] = (content_type, λ)

#     def start(self):
#         parent = self
#         class RequestHandler(BaseHTTPRequestHandler):
#             def do_GET(self):
#                 if self.path not in parent.endpoints:
#                     self.send_response(404)
#                     self.end_headers()
#                     return
#                 content_type, λ = parent.endpoints[self.path]
#                 self.send_response(200)
#                 self.send_header('Content-type', content_type)
#                 self.end_headers()
#                 self.wfile.write(λ())
#         self.web_server = HTTPServer((self.host, int(self.port)), RequestHandler)
#         def run_server():
#             self.web_server.serve_forever()
#         self.thread = threading.Thread(target=run_server, args=())
#         self.thread.start()

#     def get_host(self) -> str:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         # If the SDK is in a Docker container or on another host, it cannot reach
#         # test server via localhost/0.0.0.0/127.0.0.1/::1, so we make an external
#         # request (to NASA) to find out our external-facing IP address
#         s.connect(('192.203.230.10', 1))
#         host, port = s.getsockname()
#         return host


#     def find_port(self) -> int:
#         return random.randrange(1025,65535)

#     def stop(self):
#         if self.web_server:
#             self.web_server.shutdown()
#             self.web_server.server_close()
