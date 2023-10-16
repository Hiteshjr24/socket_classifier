import socket
import numpy as np
import cv2

# Function to send an image matrix to the server
def send_image_to_server(image_matrix, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        client_socket.send(image_matrix.tobytes())

        result_bytes = client_socket.recv(4096)  # Adjust the buffer size as needed
        result = np.frombuffer(result_bytes, dtype=np.float32)

    return result

HOST = 'localhost'  # Server's hostname or IP address
PORT = 12345        # Port used by the server

# Load an image matrix (replace with your image loading logic)
image_matrix = cv2.imread('image.jpg')

# Send the image matrix to the server for classification
result = send_image_to_server(image_matrix, HOST, PORT)

# Process the classification result as needed
print("Classification result:", result)
