import requests
import socket
import os

def check_server_status():
    """Check if the Streamlit server is running and responding"""
    try:
        # Check if we can connect to the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', 5000))
        if result == 0:
            print("✅ Port 5000 is open and accepting connections")
        else:
            print("❌ Port 5000 is not open")
        sock.close()
        
        # Try to make a request to the server
        try:
            response = requests.get("http://0.0.0.0:5000", timeout=2)
            print(f"✅ Server responded with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to server: {e}")
        
        # Display environment information
        print("\nEnvironment Information:")
        print(f"Python Version: {os.popen('python --version').read().strip()}")
        print(f"Streamlit Version: {os.popen('pip show streamlit | grep Version').read().strip()}")
        print("\nActive Processes:")
        print(os.popen('ps aux | grep streamlit').read())
        
    except Exception as e:
        print(f"Error while checking server: {e}")

if __name__ == "__main__":
    print("Debugging Streamlit Server...")
    check_server_status()