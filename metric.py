import torch
from pprint import pprint
'''
and : (1 1) -> 1, 0 otherwise
xor : (0 1) (1 0) -> 1, 0 otherwise
or : (0 0) -> 0, 1 otherwise
'''


# fucntional

def pre_processing(pred, target):
    '''
    It deletes background channel. And returns one-hot processed pred tensor.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : pred : (b,w,h,c-1), one-hot to channel
             target : (b,w,h,c-1), one-hot to channel
             eps : 1e-5
    '''
    num_classes = pred.shape[-1]
    target = target[...,1:] # select all except background class # (b,w,h,c-1)
    pred = torch.argmax(pred,-1) # (b,w,h)
    pred = torch.nn.functional.one_hot(pred, num_classes=num_classes)[...,1:] # (b,w,h,c-1) one-hot, except background class
    return pred, target, 1e-5


def get_accuracy(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    = Caution =
    Accuracy might be bigger than you expected because of large TrueNegative.
    Accuracy is not a good metric for segmentation.
    '''
    pred, target, eps = pre_processing(pred, target)
    tp_and_tn = ~torch.logical_xor(pred, target) # It returns (0,0) & (1,1) cases
    fp_and_fn = ~tp_and_tn # It returns (0,1) & (1,0) cases
    return ((torch.sum(tp_and_tn))/(torch.sum(tp_and_tn + fp_and_fn) + eps)).item()


def get_recall(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    return (torch.sum(torch.logical_and(pred, target))/(torch.sum(target) + eps)).item()


def get_precision(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    return (torch.sum(torch.logical_and(pred, target))/(torch.sum(pred) + eps)).item()


def get_f1(pred, target):
    '''
    F1 Score is equivalent to IoU.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    eps = 1e-5
    precision = get_precision(pred, target)
    recall = get_recall(pred, target)
    return 2*(precision*recall)/(precision+recall+eps)


def get_iou(pred, target):
    '''
    IoU is equivalent to F1 Score.
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)
    return (torch.sum(intersection)/(torch.sum(union)+ eps)).item()


def get_dice_coeff(pred, target):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    pred, target, eps = pre_processing(pred, target)
    intersection = torch.logical_and(pred, target) # True Positive
    return (2*(torch.sum(intersection))/(torch.sum(pred) + torch.sum(target) + eps)).item()


def get_mae(pred, target, except_background=True):
    '''
    = Input =
    pred : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, prob to channel
    target : (b,w,h,channel(=num_classes)) -> torch.FloatTensor, one-hot to channel
    = Output =
    return : some scalar
    '''
    if except_background:
        pred, target, eps = pre_processing(pred, target)
    else:
        pred = torch.argmax(pred, dim=-1)
        pred = torch.nn.functional.one_hot(pred, num_classes=target.shape[-1])
        eps = 1e-5
    bwhc = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3] # for batches, width, height and channel
    return (torch.sum(torch.abs(pred-target))).item()/(bwhc + eps)



if __name__ == '__main__':
    # dummy tensor
    channels = 3 # channels == num_classes
    pred = torch.randn((8,256,256,channels)) # prob to channels
    target = torch.randn((8,256,256,channels))
    target = torch.argmax(target,-1)
    target = torch.nn.functional.one_hot(target, num_classes=channels) # one-hot to 8 channels

    # metric
    dic = {
           'accuracy' : get_accuracy(pred, target),
           'mae' : get_mae(pred, target),
           'precision' : get_precision(pred, target),
           'recall' : get_recall(pred, target),
           'f1' : get_f1(pred, target),
           'iou' : get_iou(pred, target),
           'dice_coefficient' : get_dice_coeff(pred, target)
           }
    pprint(dic)