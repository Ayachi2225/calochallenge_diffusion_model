import torch

ckpt = torch.load('ddim3d_dataset2_electron_mean_pred.pt', map_location='cpu', weights_only=False)
print("epochs:", ckpt.get('epochs'))
print("epochs type:", type(ckpt.get('epochs')))
print("epochs shape:", ckpt.get('epochs').shape if ckpt.get('epochs') is not None else None)