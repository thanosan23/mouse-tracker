import Quartz
from ApplicationServices import AXUIElementCreateApplication, AXUIElementCopyAttributeValue, kAXErrorSuccess

def get_clickable_elements():
    clickable_elements = []
    window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    for window in window_list:
        pid = window.get('kCGWindowOwnerPID')
        if pid:
            app = AXUIElementCreateApplication(pid)
            windows, error = AXUIElementCopyAttributeValue(app, 'AXWindows', None)
            if error == kAXErrorSuccess and isinstance(windows, (list, tuple)):
                for window in windows:
                    window_info = get_window_info(window)
                    if window_info:
                        clickable_elements.append(window_info)
                    find_clickable_elements(window, clickable_elements)
            else:
                bounds = window.get('kCGWindowBounds', {})
                clickable_elements.append({
                    'role': 'Window',
                    'title': window.get('kCGWindowName', 'Untitled'),
                    'x': bounds.get('X', 0),
                    'y': bounds.get('Y', 0),
                    'width': bounds.get('Width', 0),
                    'height': bounds.get('Height', 0)
                })
    return clickable_elements

def get_window_info(window):
    try:
        position, pos_error = AXUIElementCopyAttributeValue(window, 'AXPosition', None)
        size, size_error = AXUIElementCopyAttributeValue(window, 'AXSize', None)
        title, title_error = AXUIElementCopyAttributeValue(window, 'AXTitle', None)
        if pos_error == kAXErrorSuccess and size_error == kAXErrorSuccess:
            return {
                'role': 'Window',
                'title': title if title_error == kAXErrorSuccess else 'Untitled',
                'x': position.x(),
                'y': position.y(),
                'width': size.width(),
                'height': size.height()
            }
    except Exception as e:
        print(f"Error getting window info: {e}")
    return None

def find_clickable_elements(element, clickable_elements):
    try:
        role, error = AXUIElementCopyAttributeValue(element, 'AXRole', None)
        if error == kAXErrorSuccess and role in ['AXButton', 'AXLink', 'AXRadioButton', 'AXCheckBox', 'AXPopUpButton', 'AXMenuItem']:
            position, pos_error = AXUIElementCopyAttributeValue(element, 'AXPosition', None)
            size, size_error = AXUIElementCopyAttributeValue(element, 'AXSize', None)
            title, title_error = AXUIElementCopyAttributeValue(element, 'AXTitle', None)
            if pos_error == kAXErrorSuccess and size_error == kAXErrorSuccess and position and size:
                clickable_elements.append({
                    'role': role,
                    'title': title if title_error == kAXErrorSuccess and title else 'Untitled',
                    'x': position.x(),
                    'y': position.y(),
                    'width': size.width(),
                    'height': size.height()
                })
        children, children_error = AXUIElementCopyAttributeValue(element, 'AXChildren', None)
        if children_error == kAXErrorSuccess and isinstance(children, (list, tuple)):
            for child in children:
                find_clickable_elements(child, clickable_elements)
    except Exception as e:
        print(f"Error processing element: {e}")

def main():
    elements = get_clickable_elements()
    for element in elements:
        print(f"Role: {element['role']}, Title: {element['title']}, X: {element['x']}, Y: {element['y']}, Width: {element['width']}, Height: {element['height']}")

if __name__ == "__main__":
    main()
