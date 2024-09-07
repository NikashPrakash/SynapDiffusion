import numpy as np
import requests
import time
import threading
import json

# Load data
data = np.load('data/X_eeg.npy')[47208:47308]
labels = np.load('data/y_labels.npy')[47208:47308]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MockEEGStreamer:
    def __init__(self, main_server_url, streaming_interval=1.0):
        self.main_server_url = main_server_url
        self.streaming_interval = streaming_interval
        self.is_streaming = False
        self.stream_thread = None
        self.data_index = 0

    def send_batch(self):
        if self.data_index < len(data):
            current_data = data[self.data_index]
            try:
                response = requests.post(
                    f"{self.main_server_url}/api/data",
                    json={"data": current_data},
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps({"data": current_data}, cls=NumpyEncoder)
                )
                if response.status_code == 204:
                    print(f"Data point {self.data_index} sent successfully")
                else:
                    print(f"Failed to send data point {self.data_index}. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending data point {self.data_index}: {e}")
            
            self.data_index += 1
        else:
            print("All data points sent. Stopping stream.")
            self.stop_streaming()

    def stream_data(self):
        while self.is_streaming and self.data_index < len(data):
            self.send_batch()
            time.sleep(self.streaming_interval)

    def start_streaming(self):
        if not self.is_streaming:
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self.stream_data)
            self.stream_thread.start()
            print("Streaming started")

    def stop_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            if self.stream_thread:
                self.stream_thread.join()
            print("Streaming stopped")

if __name__ == "__main__":
    MAIN_SERVER_URL = "http://172.17.87.239:9000"  # Update this with your main server's URL
    streamer = MockEEGStreamer(MAIN_SERVER_URL)

    try:
        streamer.start_streaming()
        # Let it run until all data points are sent
        while streamer.is_streaming:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        streamer.stop_streaming()