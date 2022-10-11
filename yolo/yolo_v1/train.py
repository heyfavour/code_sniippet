import os
import time

import torch

from torch.optim import SGD,Adam
from model import YOLOv1,YOLOv1ResNet,YOLOv1Loss
from data import get_dataloder
from torch.nn.utils import clip_grad_norm_



if __name__ == "__main__":
    start = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    output_path = os.path.join("output", start)
    os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # build model
    model = YOLOv1(7, 2, 20).to(device)
    train_loader, val_loader, test_loader = get_dataloder(batch_size=32)



    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = Adam(model.parameters(), lr=0.002)
    criterion = YOLOv1Loss(7, 2)

    for epoch in range(100):
        model.train()  # Set the module in training mode
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)## 7*7*(5*2+20)=1470

            # back prop
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # clip the grad
            clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            train_loss += loss.item()
            batch_time = time.time() - batch_start

            # print loss and accuracy
            if batch_idx % 10 == 0:print(f'Train [{epoch:3d}/100] Time: {batch_time} Loss: {loss.item():.6f}')



        model.eval()  # Sets the module in evaluation mode
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        print('Val set: Average loss: {:.4f}\n'.format(val_loss))
        torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))
