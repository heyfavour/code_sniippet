import torch
from torch import optim
model = None
criterion = None
optimizer = optim.Adam()

scaler = torch.cuda.amp.GradScaler()



if __name__ == '__main__':
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model()
            loss = criterion()
        scaler.scale(loss).backword()

        scaler.step(optimizer)
        scaler.update()