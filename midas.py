import torch
import torch.hub


def load_model(model_type="MiDaS_small", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    
    return midas, transform, device


def infer(midas, transform, frame, device):
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_t = transform(rgb).to(device)
    
    with torch.no_grad():
        pred = midas(input_t)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), 
            size=frame.shape[:2], 
            mode="bicubic", 
            align_corners=False
        ).squeeze()
    
    return pred.cpu().numpy()
