
import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np
import torch.nn.functional as F
from utils.utils import RunningAverage, set_logger, Params
from model import *
from data_loader import fetch_dataloader
import matplotlib.pyplot as plt

# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='data/advcifar10/resnet18', type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device=get_device()
def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40):
    loss = nn.CrossEntropyLoss()
    original_images = images.clone()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).to(images.device)
        cost.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()

    return images
def generate_and_save_adversarial_examples(loader,path):
    adv_data = []
    all_labels=[]
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels)
        adv_data.append(adv_images.cpu())
        all_labels.append(labels)

    adv_data_tensor = torch.cat(adv_data, 0)
    all_labels_tensor=torch.cat(all_labels)
    torch.save((adv_data_tensor,all_labels_tensor), path)

if __name__ == "__main__":
    # ************************** set log **************************
    set_logger(os.path.join(args.save_path, 'training.log'))

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available
    current_directory = os.getcwd()
    print(current_directory)

    for k, v in params.__dict__.items():
        logging.info('{}:{}'.format(k, v))
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)

    if params.dataset == 'cifar10':
        num_class = 10
    elif params.dataset == 'cifar100':
        num_class = 100
    else:
        num_class = 10

    logging.info('Number of class: ' + str(num_class))
    logging.info('Create Model --- ' + params.model_name)

    # ResNet 18 / 50 ****************************************
    if params.model_name == 'resnet18':
        model = ResNet18(num_class=num_class)

    elif params.model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()
    if params.cuda:
        model = model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    baseline=params.base_model
    logging.info('- Load Trained teacher model from {}'.format(baseline))
    checkpoint = torch.load(baseline,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    generate_and_save_adversarial_examples(trainloader, f'./{args.save_path}/{params.dataset}_adv_train_pre.pt')
    generate_and_save_adversarial_examples(devloader, f'./{args.save_path}/{params.dataset}_adv_test_pre.pt')