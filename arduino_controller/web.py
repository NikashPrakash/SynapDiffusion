from flask import Flask, jsonify, request
from collections import deque
import random
import torch
import numpy as np
import time

model = torch.load('path/to/model.pt')
model.eval()
app = Flask(__name__)
# TODO BUILD CLIENT ON TOP OF THIS THAT SERVES AS INTERACTIVE UI WITH ANALYTICS, SYSTEM INFO, ETC.
    # could incorporate focus tracking/personalized brain metrics like Neurable
data = deque(random.randint(0, 1) for _ in range(1000)) # TODO REPLACE THIS WITH MODEL-GENERATED SIGNALS, SHOULD PROBABLY INCLUDE TIMESTAMP/SOME WAY TO LINK SIGNAL TO ITS DATA SEGMENT
signals = []
def process_queue():
    while True:
        if not data.empty():
            eeg_data = data.popleft()
            if isinstance(eeg_data, int):
                predicted_class = eeg_data
            else:
                with torch.no_grad():
                    output = model(eeg_data)          
                predicted_class = torch.argmax(output, dim=1)
            signals.append(predicted_class)
        else:
            time.sleep(0.1)

@app.route('/api/signals', methods=['GET'])
def get_last_signal():
    print('received call')
    if signals:
        return jsonify(signal=signals.popleft())
    else:
        return jsonify(signal=None), 404

@app.route('/api/data', methods=['POST'])
def process_data():
    curr = request.json.get("data")
    if isinstance(data, int):
        data.append(curr)
        return 200
    else:
        data.append(torch.FloatTensor(np.array(curr)))
        return 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)


# def send_frame_wifi(frame_number):
#     try:
#         url = f"http://{ARDUINO_IP}/message/{frame_number}"
#         response = requests.get(url, timeout=5)
#         if response.status_code == 200:
#             print(f"Frame {frame_number} sent successfully.")
#             print("Arduino response:", response.text.strip())
#         else:
#             print(f"Failed to send frame. Status code: {response.status_code}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error sending frame: {e}")
