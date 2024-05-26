import asyncio
import json
import websockets
import torch
import numpy as np
import pickle
from ai.model import MouseModel
from config import Config

model = MouseModel(input_size=Config.INPUT_SIZE)
model.load_state_dict(torch.load('./ai/mouse_model.pth'))
model.eval()

with open('ai/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ai/output_scaler.pkl', 'rb') as f:
    output_scaler = pickle.load(f)

async def predict(websocket):
    data = await websocket.recv()
    data = json.loads(data)
    features = np.array(data['features']).astype(np.float32)
    old_features = features
    feat_shape = features.shape
    features = np.array(scaler.transform(features.reshape(-1, Config.FEATURE_SIZE)))
    features = features.reshape(feat_shape)
    features = torch.tensor(features).unsqueeze(0)
    with torch.no_grad():
        prediction = model(features.clone().detach()).numpy()
        prediction = output_scaler.inverse_transform(prediction).flatten().tolist()
    print(f"[DEBUG] Prediction: {prediction} | Coordinates: {old_features.reshape(5, -1).tolist()}")
    await websocket.send(json.dumps({'targetX': prediction[0], 'targetY': prediction[1]}))


def main():
    start_server = websockets.serve(predict, Config.HOST, Config.PORT)

    print("[DEBUG] Backend starting...")

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    main()
