# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
from torch._C import import_ir_module
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt_OR import visual_prompt
from utils.KLLoss import KLLoss
from test_swin import validate
from utils.Augmentation import *
from utils.solver_swin import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
from swin.swin_transf import SwinTransformer3D
from swin.pretrain_load import inflate_weights

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f,  Loader=yaml.FullLoader )
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    wandb.init(project=config['network']['type'],name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))

    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.


    model, clip_state_dict = clip.load(config.network.arch,device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32
    swin_model = SwinTransformer3D(patch_size = tuple(config.swin_model.patch_size), drop_path_rate = config.swin_model.drop_path_rate, in_channels = config.swin_model.in_channels, depths = tuple(config.swin_model.depths), embed_dim =config.swin_model.embed_dim, num_heads= tuple(config.swin_model.num_heads))
    # swin_model = SwinTransformer3D()
    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    swin_model =  torch.nn.DataParallel(swin_model).cuda()
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()

    wandb.watch(swin_model)
    wandb.watch(model)

    train_data = Action_DATASETS(config.data.train_list,config.data.label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS(config.data.val_list,config.data.label_list, random_shift=config.data.random_shift,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)

    # if device == "cpu":
    model_text.float()
    model_image.float()
    swin_model.float()
    # else :
    #     clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
    #     clip.model.convert_weights(model_image)
    #     clip.model.convert_weights(swin_model)


    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if config.swin_model.pretrain:
        if config.swin_model.pretrain_type == '2D_pretrain':
             pretrained_state_dict = inflate_weights(config.swin_model.pretrain,swin_model.module, tuple(config.swin_model.window_size), tuple(config.swin_model.patch_size))
             swin_model.module.load_state_dict(pretrained_state_dict, strict=False)
             del pretrained_state_dict
        else:
            unexpected_keys = ['cls_head.fc_cls.weight', 'cls_head.fc_cls.bias']
            pretrained_state_dict = torch.load(config.swin_model.pretrain, map_location='cpu')
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict['state_dict'].items() if k not in unexpected_keys}  # train的时候需要
            pretrained_state_dict = {k[9:]: v for k, v in pretrained_state_dict.items()}
            swin_model.module.load_state_dict(pretrained_state_dict, strict=False)
            del pretrained_state_dict
        print("---------------------------------------------------swin_model_load_succes----------------------------------------------")
        # print("fix_text_parameters")
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            swin_model.load_state_dict(checkpoint['swin_model'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.resume, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(train_data)

    optimizer = _optimizer(config, model, swin_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, device, model, config,num_text_aug, swin_model)
        return

    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad)) #text_encoder weight fixed
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        swin_model.train()
        for kkk,(images,list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            b,t,c,h,w = images.size()
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            # images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = texts.to(device)
            images = images.to(device).permute(0,2,1,3,4)
            # image_embedding = model_image(images)
            image_embedding = swin_model(images)


            text_embedding = model_text(texts)
            # print("*"*100)
            # print("text_embedding:{}".format(text_embedding.shape))
            # print("*"*100)

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
            
            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)
            total_loss = (loss_imgs + loss_texts)/2
            wandb.log({"train_total_loss": total_loss})
            wandb.log({"train_loss_imgs": loss_imgs})
            wandb.log({"train_loss_texts": loss_texts})

            total_loss.backward()

            optimizer.step()
            # if device == "cpu":
            #     optimizer.step()
            # else:
            #     convert_models_to_fp32(model)
            #     convert_models_to_fp32(swin_model)
            #     optimizer.step()
            #     clip.model.convert_weights(model)
            #     clip.model.convert_weights(swin_model)

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch,val_loader, classes, device, model,swin_model, config,num_text_aug)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1,best_prec1))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, model, swin_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, model, swin_model, optimizer)

if __name__ == '__main__':
    main()
