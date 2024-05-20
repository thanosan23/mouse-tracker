import asyncio
import json
import websockets
import torch
import torch.nn as nn
import numpy as np
import pickle
from ai.model import MouseModel

model = MouseModel(input_size=30)  # 5 windows * 6 features
model.load_state_dict(torch.load('./ai/mouse_model.pth'))
model.eval()

with open('ai/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

async def predict(websocket, path):
    data = await websocket.recv()
    data = json.loads(data)
    features = np.array(data['features']).astype(np.float32)
    feat_shape = features.shape
    features = np.array(scaler.transform(features.reshape(-1, 6)))
    features = features.reshape(feat_shape)
    features = torch.tensor(features).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(features.clone().detach()).numpy().flatten().tolist()
    
    await websocket.send(json.dumps({'targetX': prediction[0], 'targetY': prediction[1]}))


if __name__ == "__main__":
    start_server = websockets.serve(predict, 'localhost', 8765)

    print("[DEBUG] Backend starting...")

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()