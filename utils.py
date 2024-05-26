import torch

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    r"""
    Computes the acc@k for the specified values of k
    """
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        if k <= output.shape[1]:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(torch.zeros(1, device=target.device) - 1.0)
    return res