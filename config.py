import json
import os
import numpy as np


def load_config(path="config.json"):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_calibration(path, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q, img_size):
    np.savez(path, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, 
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, image_size=img_size)


def load_calibration(path):
    if not os.path.exists(path):
        return None
    
    data = np.load(path)
    result = {
        'K1': data['K1'], 'D1': data['D1'],
        'K2': data['K2'], 'D2': data['D2'],
        'R': data['R'], 'T': data['T'], 
        'Q': data['Q'], 
        'image_size': tuple(data['image_size'])
    }
    
    if 'R1' in data:
        result['R1'] = data['R1']
        result['R2'] = data['R2']
        result['P1'] = data['P1']
        result['P2'] = data['P2']
    
    return result
