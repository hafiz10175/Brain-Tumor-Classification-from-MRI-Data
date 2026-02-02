

import os
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
from models.forthnet50 import ForthNet50
from data.dataset import CLASS_NAMES


def get_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for ForthNet-50")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--output", type=str, default="./gradcam_output.png")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class GradCAM:
    """
    Simple Grad-CAM on the last conv layer (conv5_x / layer4)
    """

    def __init__(self, model, target_layer_name="features.7"):
        self.model = model
        self.model.eval()

        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None

        # register hooks
        target_module = dict(self.model.named_modules())[target_layer_name]
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: (1, 3, H, W)
        target_class: int or None (if None, use predicted class)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        if target_class is None:
            target_class = probs.argmax(dim=-1).item()

        score = logits[0, target_class]
        score.backward()

        # Grad-CAM: alpha_k = avg pooling of gradients over spatial dims
        gradients = self.gradients  # (N, C, H, W)
        activations = self.activations  # (N, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, probs[0].detach().cpu().numpy()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = len(CLASS_NAMES)
    model = ForthNet50(num_classes=num_classes, dropout_p=0.3).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Prepare image
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    import PIL.Image as Image
    img = Image.open(args.image).convert("L").convert("RGB")
    img_tensor = tf(img).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, target_layer_name="features.7")
    cam, probs = grad_cam.generate(img_tensor)

    pred_class = probs.argmax()
    print(f"Predicted class: {CLASS_NAMES[pred_class]} (prob={probs[pred_class]:.4f})")

    # Convert original image to numpy
    img_np = np.array(img.resize((224, 224)))
    if img_np.ndim == 2:  # grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = 0.4 * heatmap + 0.6 * img_np
    overlay = np.uint8(overlay)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM visualization to {args.output}")


if __name__ == "__main__":
    main()
