from fastai.vision import *
from fastai.tabular import *
from image_tabular.core import *
from image_tabular.dataset import *
from image_tabular.model import *
from image_tabular.metric import *
import torch
import os
import pandas as pd


def main():
    seed = 42
    img_height, img_width = 128, 128
    batch_size = 32

    # use gpu by default if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = "./data/siim-isic-melanoma-classification/"

    train_df = pd.read_csv(data_dir + "train.csv")
    test_df = pd.read_csv(data_dir + "test.csv")



    return


if __name__ == "__main__":
    main()
