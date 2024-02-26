import socket
import sys

HOST = 'localhost'  # The server's hostname or IP address
PORT = 50007        # The port used by the server

data_to_send = 'irradiancenet-s'.encode()

if data_to_send.decode() == "data":
    print('Generating data to send...')
    data_to_send = str([1.0, 1.0, 1.0] * 150).encode()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:    
    try:
        s.connect((HOST, PORT))
        s.sendall(data_to_send)
        print('Sent data:', data_to_send.decode())
        data = s.recv(1024)
        print('Received', repr(data.decode()))
    except Exception as e:
        print("Error:", e)