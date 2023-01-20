import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import draw_landmarks, draw_edge, get_train_valid_dataset, get_test_x, make_gif

import os

from glob import glob

from models import Xception

from math import sqrt

import wandb


class Trainer:

    def __init__(self,
                device,
                model,
                trainloader,
                validloader,
                test_x,
                learning_rate=0.001,
                betas=(0.5, 0.999)):
      
        self.device = device
        self.model = model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optim = Adam(self.model.parameters(), lr=learning_rate, betas=betas)

        self.trainloader = trainloader
        self.validloader = validloader

        self.train_history = []
        self.valid_history = []

        self.best_params = None
        self.best_loss = 1e18

        self.test_x = test_x.to(device)


    @torch.no_grad()
    def test(self):
        self.model.eval()

        pred = self.model(self.test_x).cpu().detach().view(-1, 68, 2)
        
        images = torch.zeros((len(self.test_x), 3, 512, 512))

        for i in range(len(self.test_x)):
            img = to_pil_image(self.test_x[i].cpu())
            img = draw_landmarks(image=img, img_type="pil", landmarks=pred[i].type(torch.int))
            img = draw_edge(image=img, img_type="pil", landmarks=pred[i].type(torch.int))

            images[i] = to_tensor(img)

        test_image = make_grid(images, nrow=int(sqrt(len(self.test_x))))
        test_image = to_pil_image(test_image)

        return test_image


    def log_history(self, save_path):
        fig, ax1 = plt.subplots()
        ax1.plot(range(len(self.train_history)), self.train_history, label="train", color="blue")

        ax2 = ax1.twiny()
        ax2.plot(range(len(self.valid_history)), self.valid_history, label="valid", color="red")
        
        plt.ylim(0, 10)
        plt.legend()
        plt.savefig(save_path)
        plt.clf()


    @torch.no_grad()
    def valid(self):
        self.model.eval()

        avg_loss = 0

        for x, y in tqdm(self.validloader):
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            avg_loss += loss.item()

            self.valid_history.append(loss.item())

            wandb.log({"valid_loss": loss.item()})

        avg_loss /= len(self.validloader)

        if avg_loss <= self.best_loss:
            self.best_loss = avg_loss
            self.best_params = self.model.state_dict()

        return avg_loss

    
    def train(self):
        self.model.train()

        avg_loss = 0

        for x, y in tqdm(self.trainloader):
            x, y = x.to(self.device), y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            avg_loss += loss.item()
            self.train_history.append(loss.item())

            wandb.log({"train_loss": loss.item()})

        avg_loss /= len(self.trainloader)

        return avg_loss
    

    def run(self, epochs, model_path="./facial_landmark_detection.pt"):

        for epoch in range(epochs):
            
            print(f"EPOCH: {epoch+1}/{epochs}")
            train_avg_loss = self.train()
            valid_avg_loss = self.valid()
            print(f"train_loss: {train_avg_loss}, valid_loss: {valid_avg_loss}, best_loss: {self.best_loss}\n\n")

            self.log_history(save_path=f"./history/loss/epoch_{epoch}.png")

            test_image = self.test()
            test_image.save(f"./history/predict/epoch_{epoch}.png")

            wandb.log({"test_image": wandb.Image(test_image)})

        torch.save(self.model.state_dict(), model_path)


if __name__ == '__main__':

    wandb.init(project="FacialLandmarkDetection", entity="donghwankim")

    wandb.run.name = "epochs50(batchnorm, repeat4)"
    wandb.save()

    args = {
        "batch_size": 16,
        "learning_rate": 0.002,
        "epochs": 50,
        "middle_repeat_n": 4
    }
    
    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Xception(middle_repeat_n=args["middle_repeat_n"]).to(device)

    trainset, validset = get_train_valid_dataset(train_ratio=0.8)

    trainloader = DataLoader(trainset, batch_size=args["batch_size"], shuffle=True)
    validloader = DataLoader(validset, batch_size=args["batch_size"], shuffle=True)

    trainer = Trainer(
        device=device,
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        test_x=get_test_x(test_n=9),
        learning_rate=args["learning_rate"]
    )

    trainer.run(
        epochs=args["epochs"]
    )

    dir = "/home/kdhsimplepro/kdhsimplepro/AI/FacialLandmarkDetection/history/predict"

    make_gif(paths=glob(os.path.join(dir, "*.png")), save_path=os.path.join(dir, "pred_log.gif"), fps=30)