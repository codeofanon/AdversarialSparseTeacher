"""Sparsity function code is referenced from:
https://github.com/HowieMa/stingy-teacher/blob/master/train_kd_stingy.py"""

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

# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='experiments/CIFAR10/adversarial_sparse_teacher/resnet18', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
torch.cuda.set_device(device_ids[0])


# ************************** training function **************************
def train_epoch(model, optim, loss_fn,adv_loss_fn, data_loader, adv_loader,params):
    model.train()
    loss_avg = RunningAverage()
    logitloss_avg = RunningAverage()
    totalloss_avg = RunningAverage()
    T = params.T

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for (train_batch, labels_batch), (adv_train_batch, adv_labels_batch) in zip(data_loader,
                                                                                    adv_loader):
            if params.cuda:
                train_batch = train_batch.cuda()
                labels_batch = labels_batch.cuda()
                adv_train_batch = adv_train_batch.cuda()
                output_batch = model(train_batch)
                ce_loss = loss_fn(output_batch, labels_batch)

                model.eval()
                with torch.no_grad():
                    adv_output_batch = model(adv_train_batch)

                    num_keep = params.sparse
                    topk_values, topk_indices = torch.topk(adv_output_batch, num_keep, dim=1)

                    sparse_logits = torch.full(adv_output_batch.shape, float("-inf")).to(adv_output_batch.device)
                    row = torch.tensor([[i] * num_keep for i in range(topk_indices.shape[0])]).to(
                        adv_output_batch.device)
                    sparse_logits[row, topk_indices] = topk_values

                model.train()
                out_probs = F.softmax(output_batch / T, dim=1)
                adv_probs = F.softmax(sparse_logits / T, dim=1)
                kl_loss = adv_loss_fn(out_probs, adv_probs)
                total = kl_loss * (params.alpha * T * T) + \
                        nn.CrossEntropyLoss()(output_batch, labels_batch) * (1. - params.alpha)
                optim.zero_grad()
                total.backward()
                optim.step()
            # update the average loss
            loss_avg.update(ce_loss.item())
            logitloss_avg.update(kl_loss.item())
            totalloss_avg.update(total.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(totalloss_avg()))
            t.update()
    return loss_avg(), logitloss_avg(), totalloss_avg()


def evaluate(model, loss_fn, data_loader, params):
    model.eval()
    # summary for current eval loop
    summ = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if params.cuda:
                data_batch = data_batch.cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean


def train_and_eval(model, optim, loss_fn,adv_loss_fn, train_loader,adversarial_loader, dev_loader,adv_dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate

    for epoch in range(params.num_epochs):
        # LR schedule *****************
        lr = adjust_learning_rate(optim, epoch, lr, params)

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info('Learning Rate {}'.format(lr))

        # ********************* one full pass over the training set *********************
        ce_loss,logit_loss,total_loss = train_epoch(model, optim, loss_fn,adv_loss_fn, train_loader,adversarial_loader, params)
        logging.info("- Cross Entropy loss : {:05.3f}".format(ce_loss))
        logging.info("- KL Div loss : {:05.3f}".format(logit_loss))
        logging.info("- Train loss : {:05.3f}".format(total_loss))


        # ********************* Evaluate for one epoch on validation set *********************
        val_metrics = evaluate(model, loss_fn, dev_loader, params)     # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)


        # save last epoch model
        save_name = os.path.join(args.save_path, 'last_model.tar')
        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
            save_name)

        # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'best_model.tar')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optim.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))


def adjust_learning_rate(opt, epoch, lr, params):
    if epoch in params.schedule:
        lr = lr * params.gamma
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr


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

    # ########################################## Dataset ##########################################
    trainloader = fetch_dataloader('train', params)
    devloader = fetch_dataloader('dev', params)
    if params.dataset=='cifar10':
        adv_train_data = torch.load('./data/advcifar10/cifar10_adv_train.pt')
        adv_test_data = torch.load('./data/advcifar10/cifar10_adv_test.pt')
    if params.dataset=='cifar100':
        if params.model_name=='resnet18':
            adv_train_data = torch.load('./data/advcifar100/resnet18/cifar100_adv_train.pt')
            adv_test_data = torch.load('./data/advcifar100/resnet18/cifar100_adv_test.pt')
        if params.model_name=='resnet50':
            adv_train_data = torch.load('./data/advcifar100/resnet18/cifar100_adv_train.pt')
            adv_test_data = torch.load('./data/advcifar100/resnet18/cifar100_adv_test.pt')

    adv_train_images, adv_train_labels = adv_train_data
    adv_train_images, adv_train_labels = adv_train_images.cpu(), adv_train_labels.cpu()
    adv_test_images, adv_test_labels = adv_test_data
    adv_test_images, adv_test_labels = adv_test_images.cpu(), adv_test_labels.cpu()

    adv_train_dataset = torch.utils.data.TensorDataset(adv_train_images, adv_train_labels)
    adv_test_dataset = torch.utils.data.TensorDataset(adv_test_images, adv_test_labels)

    advloader = torch.utils.data.DataLoader(adv_train_dataset, batch_size=params.batch_size, shuffle=False,
                                                   num_workers=params.num_workers)
    advdevloader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=params.batch_size, shuffle=False,
                                               num_workers=params.num_workers)

    # ############################################ Model ############################################
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

    # PreResNet(ResNet for CIFAR-10)  20/32 ***************
    elif params.model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif params.model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)


    elif params.model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)

    # Basic neural network ********************************
    elif params.model_name == 'net':
        model = Net(num_class, params)


    else:
        model = None
        print('Not support for model ' + str(params.model_name))
        exit()

    if params.cuda:
        model = model.cuda()

    if len(args.gpu_id) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # checkpoint ********************************
    if args.resume:
        logging.info('- Load checkpoint model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info('- Train Sparse ')

    # ############################### Optimizer ###############################
    if params.model_name == 'net' :
        optimizer = Adam(model.parameters(), lr=params.learning_rate)
        logging.info('Optimizer: Adam')
    else:
        optimizer = SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        logging.info('Optimizer: SGD')

    # ************************** LOSS **************************
    criterion = nn.CrossEntropyLoss()
    adv_criterion=nn.KLDivLoss(reduction='batchmean',log_target=True)

    # ################################# train and evaluate #################################
    train_and_eval(model, optimizer, criterion,adv_criterion, trainloader,advloader, devloader,advdevloader, params)
