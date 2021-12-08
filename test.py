# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
from re import T
from torch._C import dtype

from torch.autograd.grad_mode import no_grad
import clip
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
import numpy
from modules.Visual_Prompt_OR import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import torch.nn.functional as F
import torch.distributed as dist


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():

        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        text_embeds = text_features[torch.arange(text_features.shape[0]), text_inputs.argmax(dim=-1)]
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model.module.transformer_img(image_features).type(image_features.dtype)
            image_embeds = image_features.mean(dim=1, keepdim=False)
            # image_features = fusion_model(image_features)
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
            text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_embeds @ text_embeds.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)

        #---------------------1、vilt output----------------------
        # text_inputs = classes.to(device)
        # text_features = model.encode_text(text_inputs)
        # for iii, (image, class_id) in enumerate(tqdm(val_loader)):
        #     image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
        #     b, t, c, h, w = image.size()
        #     class_id = class_id.to(device)
        #     image_input = image.to(device).view(-1, c, h, w)
        #     image_features_in = model.encode_image(image_input).view(b, t, -1) #(96,8,512)
        #     co_similarity = []
        #     for jj  in range(text_features.shape[0]):   #一个batch  
        #         sam_text_features =  text_features[jj,:,:].repeat(image_features_in.shape[0],1,1)
        #         sam_text_inputs  =  text_inputs[jj,:].repeat(image_features_in.shape[0],1)
        #         # import pdb
        #         # pdb.set_trace()
        #         logit_per_img =  image_features_in.mean(dim=1 ,keepdim= False)  @ sam_text_features.mean(dim=1 ,keepdim= False).T
        #         logit_per_text =  sam_text_features.mean(dim=1 ,keepdim= False)  @ image_features_in.mean(dim=1 ,keepdim= False).T
        #         cls_id = torch.rand(b,1)
        #         _, sam_image_features = fusion_model(image_features_in , sam_text_features,cls_id , logit_per_img, logit_per_text)
        #         # sam_image_features = sam_image_features.mean(dim=1, keepdim = False)
        #         sam_text_features = sam_text_features[torch.arange(sam_text_features.shape[0]), sam_text_inputs.argmax(dim=-1)]
        #         sam_image_features /= sam_image_features.norm(dim=-1 , keepdim = True)
        #         sam_text_features /= sam_text_features.norm(dim=-1, keepdim = True)
        #         sam_similarity = (100 * sam_image_features.type(sam_text_features.dtype) @ sam_text_features.T)
        #         sam_similarity = torch.diag(sam_similarity)
        #         co_similarity.append(sam_similarity)
        #     similarity = (torch.stack(co_similarity)).T
        #     similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
        #     similarity = similarity.mean(dim=1, keepdim=False)
        #---------------------1、vilt output----------------------
        
        #---------------------2、创建新的similarity-matrix
            new_similarity = torch.full((image_embeds.shape[0], text_embeds.shape[0]), -100).type(similarity.dtype).to(device)
            start = 0
            end = b+1
            k_top = 3

            for i, sims in enumerate(similarity[start:end]):
                topk_sim, topk_idx = sims.topk(k = k_top,dim=0)
                encoder_output = image_features[start+i].repeat(k_top, 1,1)
                output = fusion_model.module.transformer(encoder_output, text_features[topk_idx])
                score = fusion_model.module.itm_head(output)[:,1].type(similarity.dtype).to(device) #使第一类的概率大
                new_similarity[start+ i, topk_idx] = score

        #---------------------2、创建新的similarity-matrix
            # print("similarity:{}".format(similarity))
            # print("new_similarity{}".format(new_similarity))
            values_1, indices_1 = new_similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1

    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    wandb.log({"top1": top1})
    wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                         config['data']['dataset']))
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
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data)

    best_prec1 = 0.0
    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)

if __name__ == '__main__':
    main()
