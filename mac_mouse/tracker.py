import csv
import time
from datetime import datetime
from pynput import mouse

mouse_data = []
last_mouse_position = None
last_mouse_time = None
last_target_position = (None, None)
count_prev = 0
count = 0

def on_move(x, y):
    global last_mouse_position, last_mouse_time, last_target_position, count
    current_time = time.time()
    dx = x - (last_mouse_position[0] if last_mouse_position else x)
    dy = y - (last_mouse_position[1] if last_mouse_position else y)
    dt = current_time - last_mouse_time if last_mouse_time else 0
    mouse_data.append([x, y, dx, dy, dt, 0, last_target_position[0], last_target_position[1]])
    count += 1
    last_mouse_position = (x, y)
    last_mouse_time = current_time

def on_click(x, y, button, pressed):
    global last_target_position, count, count_prev
    if pressed:
        last_target_position = (x, y)
        if mouse_data:
            for data in mouse_data[-count:]:
                count_prev = count
                data[6] = last_target_position[0]
                data[7] = last_target_position[1]
                data[5] = ((x - data[0]) ** 2 + (y - data[1]) ** 2) ** 0.5
                count = 0
        if button == mouse.Button.right:
            save_to_csv()
            print("Mouse data saved to mouse_data.csv")
            return False

def save_to_csv():
    with open('Mouse Data.csv', 'w', newline='') as csvfile:
        fieldnames = ['currentX', 'currentY', 'dx', 'dy', 'dt', 'd', 'targetX', 'targetY']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in mouse_data:
            if data[6] == '' or data[6] == None:
                continue
            writer.writerow({
                'currentX': data[0],
                'currentY': data[1],
                'dx': data[2],
                'dy': data[3],
                'dt': data[4],
                'd': data[5],
                'targetX': data[6],
                'targetY': data[7]
            })

def main():
    # Start the mouse listener
    with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
        listener.join()

if __name__ == "__main__":
    main()
