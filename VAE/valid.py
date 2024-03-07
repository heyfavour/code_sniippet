import torch
import numpy as np

from torch import nn
from model import VAE
import torchvision.transforms as transforms
from PIL import Image


def load_img(num):
    img = np.load('./data/trainingset.npy', allow_pickle=True)[num]
    img = np.transpose(img, (2, 0, 1))
    Image.fromarray(img.transpose(1, 2, 0)).show()
    img = img*2/255-1
    return torch.tensor(img,dtype=torch.float32)

def predict():
    model = VAE()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model.to(device=device)
    img = load_img(0)
    img = img.to(device).unsqueeze(0)
    mu, logvar = model.encode(img)
    decode_img = model.decode(mu).squeeze(0)
    np_img = ((decode_img+1)*255/2).cpu().detach().numpy().astype(np.uint8)
    Image.fromarray(np_img.transpose(1, 2, 0)).show()
    #sample
    decode_img = model.sample(1,device).squeeze(0)
    np_img = ((decode_img+1)*255/2).cpu().detach().numpy().astype(np.uint8)
    Image.fromarray(np_img.transpose(1, 2, 0)).show()

if __name__ == '__main__':
    predict()
