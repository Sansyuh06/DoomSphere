import cv2

mouse = {'drag': False, 'lx': 0, 'ly': 0, 'rx': -20, 'ry': 0}

def callback(event, x, y, flags, param):
    global mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse['drag'] = True
        mouse['lx'], mouse['ly'] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse['drag'] = False
    elif event == cv2.EVENT_MOUSEMOVE and mouse['drag']:
        mouse['ry'] += (x - mouse['lx']) * 0.5
        mouse['rx'] += (y - mouse['ly']) * 0.5
        mouse['lx'], mouse['ly'] = x, y

def get_rotation():
    return mouse['rx'], mouse['ry']

def reset():
    global mouse
    mouse = {'drag': False, 'lx': 0, 'ly': 0, 'rx': -20, 'ry': 0}
