import torch
from torch import optim
model = None
criterion = None
optimizer = optim.Adam()

scaler = torch.cuda.amp.GradScaler()



if __name__ == '__main__':
    for epoch in range(100):
        model.train()
        with torch.cuda.amp.autocast():
            output = model()
            loss = criterion()
        optimizer.zero_grad()
        scaler.scale(loss).backword()

        scaler.step(optimizer)
        scaler.udpate()