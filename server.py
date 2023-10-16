import socket
import cv2
import numpy as np
import threading
from model import ImageClassifier  # Import the ImageClassifier class

classifier = ImageClassifier()  # Initialize the image classifier

# Function to handle a single client
def handle_client(client_socket):
    print("Accepted connection from", client_socket.getpeername())

    image_bytes = b''
    while True:
        data = client_socket.recv(4096)
        if not data:
            break
        image_bytes += data

    image_matrix = np.frombuffer(image_bytes, dtype=np.uint8)
    image_matrix = cv2.imdecode(image_matrix, cv2.IMREAD_COLOR)

    # Classify the image matrix using the imported model
    result = classifier.classify_image(image_matrix)

    # Send the classification result back to the client
    client_socket.send(result.tobytes())

    client_socket.close()
    print("Connection closed with", client_socket.getpeername())

HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 12345      # Port used by the server

# Create a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("Server is listening for incoming connections...")

while True:
    client_socket, client_address = server_socket.accept()

    # Create a thread to handle each client
    client_handler = threading.Thread(target=handle_client, args=(client_socket,))
    client_handler.start()
