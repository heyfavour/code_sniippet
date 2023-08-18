import wandb
import random


wandb.init(project="wandb_demo", name="test_3")
wandb.config.learning_rate = 0.01
wandb.config.epoch = 100

# simulate training
offset = random.random() / 5
for epoch in range(2, wandb.config.epoch):
    for i in range(10):
        lr = lr - epoch * 0.0001
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

wandb.finish()
