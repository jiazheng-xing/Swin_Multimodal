# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR

def _optimizer(config, model, fusion_model):
    if config.solver.optim == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(),'lr': config.solver.lr },
                                {'params': model.parameters(),'lr': config.solver.lr*config.solver.f_ratio}], lr=config.solver.lr*10, betas=(0.9, 0.98), eps=1e-8,
                               weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':

        optimizer = optim.SGD([{'params': model.parameters(),'lr': config.solver.lr },
        {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
                             # config.solver.lr,
                              momentum=config.solver.momentum,
                              weight_decay=config.solver.weight_decay)
        print('SGD')
    elif config.solver.optim == 'adamw':
        vision_params = list(map(id, model.visual.parameters()))
        text_params = filter(lambda p: id(p) not in vision_params,
                             model.parameters())

        # fc_params_id = list(map(id, fusion_model.module.fc.parameters()))
        # base_params = filter(lambda p: id(p) not in fc_params_id, fusion_model.module.parameters())

        # optimizer = optim.AdamW([{'params': text_params},
        #                          {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
        #                          {'params': base_params, 'lr': config.solver.lr * config.solver.f_ratio},
        #                          {'params': fusion_model.module.fc.parameters(), 'lr': config.solver.lr * config.solver.f_ratio}],
        #                         betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
        #                         weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        optimizer = optim.AdamW([{'params': text_params, 'lr': config.solver.lr * config.solver.ratio},
                                 {'params': model.visual.parameters(), 'lr': config.solver.lr * config.solver.ratio},
                                 {'params': fusion_model.parameters(), 'lr': config.solver.lr * config.solver.f_ratio},],
                                betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
                                weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        # optimizer = optim.AdamW([{'params': text_params},
        #                          {'params': model.visual.parameters(), 'lr':config.solver.lr},
        #                          {'params': base_params, 'lr': config.solver.lr* config.solver.f_ratio* config.solver.f_ratio},
        #                          {'params':  fusion_model.module.fc.parameters(), 'lr':  config.solver.lr * config.solver.f_ratio}],
        #                         betas=(0.9, 0.98), lr=config.solver.lr, eps=1e-8,
        #                         weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])

        # optimizer = optim.AdamW([{'params': model.parameters(),'lr': config.solver.lr },],
        #                         betas=(0.9, 0.98), lr=config.solver.lr *  config.solver.f_ratio, eps=1e-8,
        #                         weight_decay=config.solver.weight_decay)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # for param_group in optimizer.param_groups:
        #     print(param_group['lr'])


        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer

def _lr_scheduler(config,optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler