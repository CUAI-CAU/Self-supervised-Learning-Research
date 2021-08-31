import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .misc import AverageMeter, accuracy


def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter() ## TODO
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        ## TODO
#         for batch_idx, (inputs, targets, _, _) in enumerate(test_loader):
        for batch_idx, (inputs, targets) in enumerate(test_loader) : 
            
#             print('>> check input shape :', inputs.shape)
#             print('>> check target shape :', targets.shape)
            
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            ## TODO : 여긴 클래스 수따라 달라짐
            prec1 = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1, inputs.shape[0])
            #top2.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}.".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    #top2=top2.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    return losses.avg, top1.avg