import torch
from load_model.pytorch_loader import load_pytorch_model

model = load_pytorch_model('models/face_mask_detection.pth')

def convert(image):

    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("models/face_mask_detection_libtorch.pth")


if __name__ == "__main__":

    convert(torch.rand(2, 3, 500,500)

