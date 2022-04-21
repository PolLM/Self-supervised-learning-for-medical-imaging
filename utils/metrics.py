#%%
import torch
import numpy as np

def accuracy_batch(labels, outputs):
    '''
    Computes the accuracy over one batch
    '''
    total_acc = []
    for lab, out in zip(labels, outputs):
        lab, out = torch.tensor(lab), torch.tensor(out)
        preds = out.argmax(-1)
        acc = (preds == lab.view_as(preds)).float().detach().numpy().mean()
        total_acc.append(acc)
    return np.mean(total_acc)

def accuracy(labels, outputs_logits):
    '''
    Computes the accuracy
    '''
    outputs = torch.softmax(outputs_logits, dim=1)
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

def topk_accuracy(output, target, topk=(1,)):
    '''
    Computes the accuracy over the k top predictions for the specified values of k
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    labels = torch.rand(10)
    outputs = torch.rand((10, 4))

    acc = accuracy(labels, outputs)

# %%
