import torch


def set_optimizer(model, optim, lr):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        elif 'conv.weight' in k:
            pg1.append(v)  # apply weight_decay
        elif 'position_mixing' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if optim == 'Adam':
        optimizer = torch.optim.Adam(pg0, lr=lr, betas=(0.937, 0.999))  # adjust beta1 to momentum
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(pg0, lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(pg0, lr=lr, momentum=0.937, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005 if optim != 'AdamW' else 0.01})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print(optimizer)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    return optimizer
