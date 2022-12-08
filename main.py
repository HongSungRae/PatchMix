# library
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import sys
import argparse
import segmentation_models_pytorch as smp
import wandb
import json
from pprint import pprint
import random
from multiprocessing import Manager

# local
from utils import *
from datasets.busi import BUSI
from datasets.seegene import Seegene
from datasets.digestpath19 import DigestPath19
from metric import *
from parameters import SEED

# seed
# random.seed(SEED)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    # Comment
    parser.add_argument('--comment','--c', default='COMMENT', type=str,
                        help='Any comment?')
    
    # configurations
    parser.add_argument('--experiment_name','--e', default='NAME', type=str,
                        help='experimental name')
    parser.add_argument('--save_path', default='./exp', type=str,
                        help='save path')
    parser.add_argument('--model', default='unet', type=str,
                        choices=['unet', 'unet++', 'deeplabv3', 'deeplabv3+'])                    
    parser.add_argument('--backbone', default='resnet34', type=str,
                        help='backbone network for segmentation network')
    parser.add_argument('--dataset', default='busi', type=str,
                        help='실험할 데이터셋', choices=['busi','seegene','digestpath19'])
    parser.add_argument('--size', default=256, type=int,
                        help='size of image')
    parser.add_argument('--busi_what', default='benign', type=str,
                        help='BUSI dataset에서 어떤 데이터를 사용할지 설정', choices=['benign', 'malignant', 'normal'])
    parser.add_argument('--seegene_what', default='NM', type=str,
                        help='Seegene dataset에서 어떤 데이터를 사용할지 설정', choices=['NM','M'])
    parser.add_argument('--augmentation', '--a', default=None,
                        help='augmentation', choices=[None, 'image_aug', 
                                                     'cutmix_half', 'cutmix_dice', 'cutmix_random', 'cutmix_random_simple', 'cutmix_random_gaussian', 'cutmix_random_tumor', 'cutmix_random_poisson',
                                                     'cp_naive', 'cp_simple', 'cp_poisson', 'cp_gaussian', 'cp_tumor'])
    parser.add_argument('--aug_p', default=0.3, type=float,
                        help='augmentation prop')
    parser.add_argument('--ratio', default=None, type=float,
                        help='Ratio of [original train data] : [original train data + aug data]')                       
    parser.add_argument('--batch_size', '--bs', default=64, type=int,
                        help='batch size')
    parser.add_argument('--optim', default='adam', type=str,
                        help='optimizer', choices=['sgd','adam','adagrad'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay', '--ld', default=1e-3, type=float,
                        help='learning rate decay')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        help='weight_decay')
    parser.add_argument('--epochs', default=50, type=int,
                        help='train epoch')

    # GPU srtting							
    parser.add_argument('--gpu_id', default='1', type=str,
                        help='How To Check? : cmd -> nvidia-smi')
    
    # For test only
    parser.add_argument('--test_only', action='store_true',
                        help='How To Make TRUE? : --test_only, Flase : default')
    parser.add_argument('--test_path', default='', type=str,
                        help='테스트할 model file있는 path')

    args = parser.parse_args()
    return parser.parse_known_args()[0] if known else args




def train(model, train_loader, criterion, optimizer, epoch, num_epoch):
    model.train()
    train_loss = AverageMeter()
    for i, (image, mask) in enumerate(train_loader):
        mask = torch.einsum('bwhc->bcwh', mask)
        image, mask = image.cuda(), mask.cuda()
        mask_pred = model(image)

        loss = criterion(mask_pred, mask)
        train_loss.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10 == 0 and i!= 0:
            print(f'Epoch : [{epoch}/{num_epoch}] [{i}/{len(train_loader)}] [Train Loss : {loss:.4f}]')
    
    wandb.log({'train_loss':train_loss.avg},step=epoch)
    



def validation(model, validation_loader, criterion, epoch):
    model.eval()
    validation_loss = AverageMeter()
    with torch.no_grad():
        for i, (image, mask) in enumerate(validation_loader):
            mask = torch.einsum('bwhc->bcwh', mask)
            image, mask = image.cuda(), mask.cuda()
            pred_mask = model(image)

            loss = criterion(pred_mask, mask).item()
            validation_loss.update(loss)
    print(f'\nValidation | Epoch : {epoch} | Loss : {validation_loss.avg}')
    wandb.log({'validation_loss':validation_loss.avg}, step=epoch)

    return validation_loss.avg



def test(model, test_loader, save_path, args):
    print("=================== Test Start ====================")
    
    # metrics
    precision_av = AverageMeter()
    recall_av = AverageMeter()
    f1_av = AverageMeter()
    iou_av = AverageMeter()
    dice_coeff_av = AverageMeter()
    mae_av = AverageMeter()
    accuracy_av = AverageMeter()
    
    # model test
    model.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image, mask = image.cuda(), mask.cuda()
            pred_mask = model(image)
            pred_mask = torch.einsum('bchw->bhwc', pred_mask)

            precision_av.update(get_precision(pred_mask, mask))
            recall_av.update(get_recall(pred_mask, mask))
            f1_av.update(get_f1(pred_mask, mask))
            iou_av.update(get_iou(pred_mask, mask))
            dice_coeff_av.update(get_dice_coeff(pred_mask, mask))
            mae_av.update(get_mae(pred_mask, mask))
            accuracy_av.update(get_accuracy(pred_mask, mask))
    
    result = {
              'Precision' : f'{precision_av.avg:.3f}+-{precision_av.std:.3f}',
              'Recall' : f'{recall_av.avg:.3f}+-{recall_av.std:.3f}',
              'F1' : f'{f1_av.avg:.3f}+-{f1_av.std:.3f}',
              'MAE' : f'{mae_av.avg:.3f}+-{mae_av.std:.3f}',
              'IoU' : f'{iou_av.avg:.3f}+-{iou_av.std:.3f}',
              'Dice_Coefficient' : f'{dice_coeff_av.avg:.3f}+-{dice_coeff_av.std:.3f}',
              'Accuracy' : f'{accuracy_av.avg:.3f}+-{accuracy_av.std:.3f}'
              }

    # Save result, confusion matrix
    with open(save_path + '/result.json', 'w') as f:
        json.dump(result, f, indent=2)
    wandb.log({"Precision" : precision_av.avg,
               "Recall" : recall_av.avg,
               "F1" : f1_av.avg,
               "MAE" : mae_av.avg,
               "IoU" : iou_av.avg,
               "Dice Coefficient" : dice_coeff_av.avg,
               "Accuracy" : accuracy_av.avg})
    # wandb.save(save_path + '/result.json')
    wandb.alert(title=f'실험 완료 : {args.experiment_name}',
                text=json.dumps(result),
                level= wandb.AlertLevel.INFO)

    pprint(result)
    print("=================== Test End ====================")



def main(args):
    # initial settings
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # log
    wandb.login()
    wandb.init(project="patchmix", entity="hong", name=args.experiment_name)
    wandb.config.update(args)
    start = time.time()

    # Save path
    if args.experiment_name == 'NAME':
        save_path = os.path.join(args.save_path, str(start).split('.')[-1])
    else:
        save_path = os.path.join(args.save_path, args.experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save configuration
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # dataset and dataloader
    manager = Manager()
    img_cache = manager.dict()
    # if args.ratio != None:
    #     args.aug_p = 0.0
    if args.dataset == 'busi':
        num_classes = 2
        train_dataset = BUSI(split='train',
                             dataset = args.busi_what, 
                             augmentation=args.augmentation, 
                             aug_p=args.aug_p,
                             ratio=args.ratio, 
                             size=args.size)
        validation_dataset = BUSI(split='validation', 
                                  dataset = args.busi_what, 
                                  augmentation=None,
                                  size=args.size)
        test_dataset = BUSI(split='test', 
                            dataset = args.busi_what, 
                            augmentation=None, 
                            size=args.size)
        # if args.ratio != None:
        #     additional_train_dataset = BUSI(split='train', 
        #                                     dataset = args.busi_what, 
        #                                     augmentation=args.augmentation, 
        #                                     aug_p=1.0,
        #                                     ratio=args.ratio, 
        #                                     size=args.size)
        #     additional_train_dataset.length = int(len(train_dataset) * args.ratio)
        #     train_dataset += additional_train_dataset
    elif args.dataset == 'seegene':
        num_classes = 2
        train_dataset = Seegene(split='train',
                                cache=img_cache,
                                dataset = args.seegene_what,
                                augmentation=args.augmentation,
                                aug_p=args.aug_p,
                                ratio=args.ratio,
                                size=args.size)
        validation_dataset = Seegene(split='validation',
                                     cache=img_cache,
                                     dataset = args.seegene_what,
                                     augmentation=None,
                                     size=args.size)
        test_dataset = Seegene(split='test',
                               cache=img_cache,
                               dataset = args.seegene_what,
                               augmentation=None,
                               size=args.size)
        # if args.ratio != None:
        #     additional_train_dataset = Seegene(split='train',
        #                                        dataset = args.seegene_what,
        #                                        augmentation=args.augmentation,
        #                                        aug_p=1.0,
        #                                        ratio=args.ratio,
        #                                        size=args.size)
        #     additional_train_dataset.length = int(len(train_dataset) * args.ratio)
        #     train_dataset += additional_train_dataset
    elif args.dataset == 'digestpath2019':
        num_classes = 2
        raise NotImplementedError('digestpath2019')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f'=== DataLoader | (Train/Val/Test) : ({len(train_dataset)},{len(validation_dataset)},{len(test_dataset)}) ===')

    # model load
    if args.model == 'deeplabv3':
        model = smp.DeepLabV3(encoder_name=args.backbone,classes=num_classes).cuda()
    elif args.model == 'deeplabv3+':
        model = smp.DeepLabV3Plus(encoder_name=args.backbone,classes=num_classes).cuda()
    elif args.model == 'unet':
        model = smp.Unet(encoder_name=args.backbone,classes=num_classes).cuda()
    elif args.model == 'unet++':
        model = smp.UnetPlusPlus(encoder_name=args.backbone,classes=num_classes).cuda()

    # define criterion
    criterion = nn.BCEWithLogitsLoss().cuda()

    # define Optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    milestones = [int(args.epochs/3),int(args.epochs/2)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)


    # Train, Validation, Test
    if args.test_only: # test only
        with open(os.path.join(args.test_path,'configuration.json'), 'r') as f:
            configuration = json.load(f)
        model_name = configuration['model']
        if model_name == 'deeplabv3':
            model = smp.DeepLabV3(encoder_name=configuration['backbone'],classes=2).cuda()
        elif model_name == 'deeplabv3+':
            model = smp.DeepLabV3Plus(encoder_name=configuration['backbone'],classes=2).cuda()
        elif model_name == 'unet':
            model = smp.Unet(encoder_name=configuration['backbone'],classes=2).cuda()
        elif model_name == 'unet++':
            model = smp.UnetPlusPlus(encoder_name=args.backbone,classes=2).cuda()
        model.load_state_dict(torch.load(os.path.join(args.test_path, 'model.pth')))
        test(model, test_loader, './exp/'+args.experiment_name, args)
    else: # train & validation
        wandb.watch(model, criterion, log="all", log_freq=5)
        global best_loss, count
        best_loss = 100000.0
        count = 0
        for epoch in tqdm(range(1,args.epochs+1)):
            train(model, train_loader, criterion, optimizer, epoch, args.epochs)
            validation_loss = validation(model, validation_loader, criterion, epoch)
            scheduler.step()

            # Early Stop
            if best_loss >= validation_loss:
                best_loss = validation_loss
                count = 0
            else:
                count += 1
            
            if (epoch >= 30) and (count >= 3):
                print(f'Early stopping at [{epoch}]epoch')
                torch.save(model.state_dict(), f'{save_path}/model.pth')
                break

            # save best model
            if epoch == args.epochs:
                torch.save(model.state_dict(), f'{save_path}/model.pth')
                # wandb.save(f'{save_path}/model.pth')

        # test
        del manager, img_cache, train_dataset, train_loader, validation_dataset, validation_loader
        manager = Manager()
        img_cache = manager.dict()
        test_dataset.cache = img_cache
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test(model, test_loader, save_path, args)

    print(f"Process Complete : it took {((time.time()-start)/60):.2f} minutes")

if __name__ == '__main__':
    args = parse_opt()
    main(args)