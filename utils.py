import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image

from PIL import Image, ImageDraw

import numpy as np

import cv2

from tqdm import tqdm

from random import choices

from detect import detect_landmarks_dlib



def to_np(image, img_type):
    if img_type == "pil":
        img = to_tensor(image.convert("RGB"))
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    elif img_type == "np":
        img = image.copy()

    elif img_type == "tensor":
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return img


def to_pil(image, img_type):
    if img_type == "np":
        img = to_pil_image(torch.from_numpy(image/255).permute(2, 0, 1))

    elif img_type == "pil":
        img = image.copy()

    elif img_type == "tensor":
        img = to_pil_image(image)

    return img


def to_tensor_image(image, img_type, device):
    
    if img_type == "pil":
        img = to_tensor(image.resize((512, 512)).convert("RGB")).to(device)

    elif img_type == "tensor":
        img = image.to(device)
    
    elif img_type == "np":
        img = torch.from_numpy(cv2.resize(image, dsize=(512, 512)) / 255).permute(2, 0, 1).type(torch.float32).to(device)
    
    return img


def draw_landmarks(image, img_type="pil", landmarks=None):

    if img_type == "np":
        img = to_pil_image(torch.from_numpy(image/255).permute(2, 0, 1))

    elif img_type == "pil":
        img = image.copy()

    elif img_type == "tensor":
        img = to_pil_image(image)

    draw = ImageDraw.Draw(img)
    
    for x, y in landmarks:
        draw.point((x, y), fill=(256, 154, 69))

    if img_type == "np": return (to_tensor(img).permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    elif img_type == "pil": return img
    elif img_type == "tensor": return to_tensor(img)


def draw_edge(image, img_type="pil", landmarks=None):
    edge_point = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1,
        18, 19, 20, 21, -1,
        23, 24, 25, 26, -1,
        28, 29, 30, -1,
        32, 33, 34, 35, -1,
        37, 38, 39, 40, 41, 36,
        43, 44, 45, 46, 47, 42,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48,
        61, 62, 63, 64, 65, 66, 67, 60
    ]

    if img_type == "pil":
        img = image.copy()

    elif img_type == "np":
        img = to_pil_image(torch.from_numpy(image/255).permute(2, 0, 1))

    elif img_type == "tensor":
        img = to_pil_image(image)
        
    draw = ImageDraw.Draw(img)

    for i in range(len(landmarks)):
        if edge_point[i] != -1:
            x1, y1 = landmarks[i]
            x2, y2 = landmarks[edge_point[i]]

            if (x1, y1) == (0, 0) or (x2, y2) == (0, 0):
                continue
                
            draw.line((x1, y1, x2, y2), fill=(245, 154, 69), width=2)
    
    if img_type == "pil": return img
    elif img_type == "np": return (to_tensor(img).permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    elif img_type == "tensor": return to_tensor(img)


class LandmarkDataset(Dataset):

    def __init__(self, img_number_list):
        super().__init__()

        self.img_number_list = img_number_list
        self.root_dir = "/home/kdhsimplepro/kdhsimplepro/AI/"

    def __getitem__(self, index):
        number = self.img_number_list[index]
        img = to_tensor(Image.open(os.path.join(self.root_dir, "ffhq512", f"{number}.png")).convert("RGB"))
        landmarks = torch.load(os.path.join(self.root_dir, "FacialLandmarkDetection", "ffhq_landmarks", f"{number}.pt"))

        return img, landmarks

    def __len__(self):
        return len(self.img_number_list)


def get_train_valid_dataset(train_ratio=0.8):

    ffhq_paths = glob.glob('/home/kdhsimplepro/kdhsimplepro/AI/FacialLandmarkDetection/ffhq_landmarks/*.pt')
    img_number_list = [path.split("/")[-1].split(".pt")[0] for path in ffhq_paths]

    train_len = int(len(img_number_list) * train_ratio)
    
    trainset = LandmarkDataset(img_number_list=img_number_list[:train_len])
    validset = LandmarkDataset(img_number_list=img_number_list[train_len:])

    return trainset, validset


def get_test_x(test_n):
    ffhq_paths = glob.glob('/home/kdhsimplepro/kdhsimplepro/AI/FacialLandmarkDetection/ffhq_landmarks/*.pt')
    img_number_list = [path.split("/")[-1].split(".pt")[0] for path in ffhq_paths]

    img_number_list = choices(img_number_list, k=test_n)

    test_x = torch.zeros((test_n, 3, 512, 512))

    for i, img_number in enumerate(img_number_list):
        img = to_tensor(Image.open(os.path.join("/home/kdhsimplepro/kdhsimplepro/AI/ffhq512/", f"{img_number}.png")).convert("RGB"))
        test_x[i] = img

    return test_x


def make_gif(paths, save_path, fps=500):
    img, *imgs = [Image.open(path) for path in paths]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=1)


if __name__ == '__main__':

    # ---------------- make dataset --------------------------
    root_path = '/home/kdhsimplepro/kdhsimplepro/AI/'

    ffhq_dir = os.path.join(root_path, "ffhq512")
    ffhq_paths = glob.glob(os.path.join(ffhq_dir, "*.png"))
    img_number_list = [path.split("/")[-1].split(".png")[0] for path in ffhq_paths]

    error_img_number = []
    
    for path, img_number in tqdm(zip(ffhq_paths, img_number_list)):
        image = Image.open(path)
        landmarks = detect_landmarks_dlib(image, img_type="pil")

        if landmarks:
            landmarks = landmarks[0].view(-1).type(torch.float32)
            torch.save(landmarks, f"ffhq_landmarks/{img_number}.pt")

        else:
            error_img_number.append(img_number)

    ## ---------------------- dataset test -----------------------------
    # root_path = '/home/kdhsimplepro/kdhsimplepro/AI/'

    # ffhq_dir = os.path.join(root_path, "FacialLandmarkDetection", "ffhq_landmarks")
    # ffhq_paths = glob.glob(os.path.join(ffhq_dir, "*.pt"))
    # img_number_list = [path.split("/")[-1].split(".pt")[0] for path in ffhq_paths]

    # img = Image.open(ffhq_paths[0])
    # landmarks = torch.load(f"ffhq_landmarks/{img_number_list[0]}.pt").view(68, 2)

    # img = draw_landmarks(image=img, img_type="pil", landmarks=landmarks)
    # img.show()

    pass