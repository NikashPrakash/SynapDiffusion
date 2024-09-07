from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from collections import deque
import torch
import numpy as np
import time
import threading
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import src.models as models

# Load the model
model = models.D_MI_WaveletCNN()
model.load_state_dict(torch.load('model.pt')[0])
model.eval()

app = Flask(__name__)
socketio = SocketIO(app)

data_queue = deque(maxlen=1000)  # Enforce length for thread-safety
signals = deque(maxlen=1000)

def process_queue():
    while True:
        if data_queue:
            eeg_data = data_queue.popleft()
            if isinstance(eeg_data, int):
                predicted_class = eeg_data
            else:
                with torch.no_grad():
                    output = model(eeg_data.unsqueeze(0))  # Add batch dimension
                    predicted_class = torch.argmax(output, dim=1).item()
            signals.append(predicted_class)
        else:
            time.sleep(0.1)

@app.route('/api/signals', methods=['GET'])
def get_last_signal():
    print('balls')
    if signals:
        signal = signals.popleft()
        return jsonify(signal)
    else:
        return jsonify(signal=None), 404

@app.route('/api/data', methods=['POST'])
def process_data():
    curr = request.json.get("data")
    if isinstance(curr, int):
        data_queue.append(curr)
    else:
        data_queue.append(torch.FloatTensor(np.array(curr)))
    return '', 204  # No content response

@socketio.on('eeg_data')
def handle_eeg_data(data):
    try:
        eeg_data = np.array(data)
        data_queue.append(torch.FloatTensor(eeg_data))
    except Exception as e:
        print(f"Error processing WebSocket data: {e}")

if __name__ == '__main__':
    # Start the queue processing in a separate thread
    processing_thread = threading.Thread(target=process_queue, daemon=True)
    processing_thread.start()

    # Run the Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=9000, debug=True)