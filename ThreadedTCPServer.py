import socket
import threading
import socketserver
from time import gmtime, strftime, sleep
import _thread

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        print("got request: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        sleep(5)
        print("send response: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass

def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(bytes(message, 'ascii'))
        response = str(sock.recv(1024), 'ascii')
        print("Received: {}".format(response))

if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 0

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        ip, port = server.server_address

        # Start a thread with the server -- that thread will then start one
        # more thread for each request
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)

        _thread.start_new_thread(client, (ip, port, "Hello World 1"))
        _thread.start_new_thread(client, (ip, port, "Hello World 2"))
        _thread.start_new_thread(client, (ip, port, "Hello World 3"))

        sleep(6)

        print("Shutting down")

        server.shutdown()