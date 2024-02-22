import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from model import UNet


def deal_data(file, scale=0.5):
    img = Image.open(file)
    w, h = img.size
    newW, newH = int(scale * w), int(scale * h)
    img = img.resize((newW, newH), resample=Image.BICUBIC)

    img = np.asarray(img).transpose((2, 0, 1))  # [w h c]->[c w h]
    return img, w, h


def mask_to_img(mask):
    mask = mask.argmax(dim=1)
    mask = mask[0].long().squeeze().numpy()
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    for i, v in enumerate([0, 1]):
        out[mask == i] = v
    return Image.fromarray(out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=2)
    # model.load_state_dict(torch.load("./unet_carvana_scale0.5_epoch2.pth"))
    model.load_state_dict(torch.load("./best_21.pth",map_location=device))
    model.to(device=device)
    input = "./data/valid/image.jpg"
    output = "./data/valid/output.jpg"

    img, w, h = deal_data(input)
    with torch.no_grad():
        img = torch.as_tensor(img.copy()).unsqueeze(0).float().contiguous().to(device)
        mask = model(img).cpu()
        # mask = F.interpolate(mask, (h, w), mode='bilinear')
        mask = mask_to_img(mask)
        mask.save(output)
