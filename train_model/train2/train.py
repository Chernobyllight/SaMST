import os
project_root = os.path.abspath('../..')
import sys
sys.path.append(project_root)



import random

import yaml
import os
import sys
import numpy as np
import time

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

from networks.transfer_net import TransformerNet
from loss.vgg import Vgg16
from train_model import utils

def check_paths(opt):
    try:
        if not os.path.exists(opt['save_model_dir']):
            os.makedirs(opt['save_model_dir'])

        if opt['checkpoint_model_dir'] is not None and not (os.path.exists(opt['checkpoint_model_dir'])):
            os.makedirs(opt['checkpoint_model_dir'])
    except OSError as e:
        print(e)
        sys.exit(1)


def train(opt):
    device = torch.device("cuda" if opt['cuda'] else "cpu")

    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])

    transform = transforms.Compose([
        transforms.Resize(opt['image_size']),  # the shorter side is resize to match image_size
        transforms.CenterCrop(opt['image_size']),
        transforms.ToTensor(),  # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255))  # convert back to [0, 255]
    ])
    train_dataset = datasets.ImageFolder(opt['dataset'], transform)
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)  # to provide a batch loader

    style_image = [f for f in os.listdir(opt['style_image'])]
    style_num = len(style_image)
    print('total style number:',style_num)

    labels = [i for i in range(0,style_num+1)]
    labels = torch.Tensor(labels).cuda()


    transformer = TransformerNet(style_num=style_num)
    print('# MODEL parameters:', sum(param.numel() for param in transformer.parameters()), '\n')
    begin_epoch = 0
    if opt['begin_checkpoint'] is not None:
        state_dict = torch.load(opt['begin_checkpoint'])
        transformer.load_state_dict(state_dict)
        print("load checkpoint model to train")
        begin_epoch = opt['begin_epoch']
    transformer = transformer.to(device)


    optimizer = Adam(transformer.parameters(), opt['lr'])

    mse_loss = torch.nn.MSELoss()


    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.Resize(opt['style_size']),
        transforms.CenterCrop(opt['style_size']),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_batch = []

    for i in range(style_num):
        style = utils.load_image(opt['style_image'] + style_image[i], size=opt['style_size'])
        style = style_transform(style)
        style_batch.append(style)

    style = torch.stack(style_batch).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]


    content_weight = float(opt['content_weight'])
    style_weight = float(opt['style_weight'])
    ae_weight = float(opt['ae_weight'])

    total_epochs = opt['epochs']
    for e in range(begin_epoch + 1,total_epochs+1):

        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_ae_loss = 0.

        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)

            if n_batch < opt['batch_size']:
                break  # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            optimizer.zero_grad()  # initialize with zero gradients

            batch_style_id = [random.randint(1,style_num) for i in range(count - n_batch, count)]


            for i in range(n_batch):
                batch_style_id.append(0)

            x = x.repeat(2,1,1,1)
            # print(batch_style_id)
            y,embedding = transformer(x.to(device), style_id=batch_style_id)


            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)


            y = torch.split(y, n_batch , dim=0)  # 按照4这个维度去分，每大块包含2个小块
            y1 = y[0]
            y2 = y[1]

            x = torch.split(x, n_batch , dim=0)
            x1 = x[0]
            x2 = x[1]


            features_y = vgg(y1.to(device))
            features_x = vgg(x1.to(device))

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            batch_style_id = batch_style_id[0:n_batch]
            for i in range(len(batch_style_id)):
                batch_style_id[i] -= 1
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[batch_style_id, :, :])
            style_loss *= style_weight

            ae_loss = ae_weight * mse_loss(y2.to(device),x2.to(device))


            total_loss = content_loss + style_loss + ae_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_ae_loss += ae_loss.item()

            if (batch_id + 1) % opt['log_interval'] == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\tae: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                    agg_ae_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if opt['checkpoint_model_dir'] is not None and (batch_id + 1) % opt['checkpoint_interval'] == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(opt['checkpoint_model_dir'], ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

        if e % opt['save_interval'] == 0:
            # save model
            transformer.eval().cpu()
            save_model_filename = "epoch_" + str(e) + ".model"
            save_model_path = os.path.join(opt['save_model_dir'], save_model_filename)
            torch.save(transformer.state_dict(), save_model_path)
            print("\ntrained model saved at", save_model_path)
            transformer.to(device).train()


        if e % opt['step_size'] == 0:
            lr = opt['lr'] * (opt['weight_decay'] ** (e // opt['step_size']))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('now learning rate: ',optimizer.state_dict()['param_groups'][0]['lr'])





def main():

    with open('train.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)

    random.seed(7)
    check_paths(opt)
    train(opt)


if __name__ == "__main__":
    main()




