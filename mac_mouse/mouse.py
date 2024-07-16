from pynput.mouse import Controller
from time import sleep, time
import asyncio
import json
import websockets
from websockets.exceptions import ConnectionClosedOK

HOST = 'localhost'
PORT = 8765
WINDOW_SIZE = 5
DEBUG = False

mouse_data = []
last_mouse_position = None
last_mouse_time = None
mouse_controller = Controller()

def capture_mouse_position():
    global last_mouse_position, last_mouse_time
    current_time = time()
    currentX, currentY = mouse_controller.position
    dx = currentX - (last_mouse_position[0] if last_mouse_position else currentX)
    dy = currentY - (last_mouse_position[1] if last_mouse_position else currentY)
    dt = current_time - last_mouse_time if last_mouse_time else 0
    mouse_data.append([currentX, currentY, dx, dy, dt])
    last_mouse_position = (currentX, currentY)
    last_mouse_time = current_time

async def send_mouse_data():
    while True:
        try:
            async with websockets.connect(f'ws://{HOST}:{PORT}/predict') as websocket:
                while True:
                    if len(mouse_data) > WINDOW_SIZE:
                        recent_data = mouse_data[-WINDOW_SIZE:]
                        features = [item for sublist in recent_data for item in sublist]
                        print(features)
                        await websocket.send(json.dumps({'features': features}))

                        response = await websocket.recv()
                        prediction = json.loads(response)
                        if DEBUG:
                            print(f"[DEBUG] Prediction: {prediction}")
                        
                        mouse_controller.position = (prediction['targetX'], prediction['targetY'])

                    await asyncio.sleep(0.1)  # Ensure the loop yields control
        except ConnectionClosedOK:
            print("Connection closed normally. Reconnecting...")
            await asyncio.sleep(0.1)  # Wait a bit before reconnecting

async def capture_mouse_positions_periodically():
    while True:
        capture_mouse_position()
        await asyncio.sleep(0.001)  # Adjust the sleep time as needed

async def main():
    await asyncio.gather(
        capture_mouse_positions_periodically(),
        send_mouse_data()
    )

if __name__ == "__main__":
    asyncio.run(main())
