from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS module
import torch
import torch.nn as nn
import numpy as np
import pickle

class MouseModel(nn.Module):
    def __init__(self, input_size):
        super(MouseModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

app = Flask(__name__)
CORS(app) 

model = MouseModel(input_size=30)  # 5 windows * 6 features
model.load_state_dict(torch.load('ai/mouse_model.pth'))
model.eval()

# Load the scaler
with open('ai/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).astype(np.float32)
    feat_shape = features.shape
    features = np.array(scaler.transform(features.reshape(-1, 6)))
    features = features.reshape(feat_shape)
    features = torch.tensor(features).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(torch.tensor(features)).numpy().flatten().tolist()
    return jsonify({'targetX': prediction[0], 'targetY': prediction[1]})

if __name__ == '__main__':
    app.run(debug=True)