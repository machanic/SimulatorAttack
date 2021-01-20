import torch
import torch.nn as nn



def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def xent_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.003,
              epsilon=0.031,
              perturb_steps=10,
              distance='l_inf'):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = criterion(model(x_adv), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = x_adv.detach()
    x_adv.requires_grad = False
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    loss_robust = criterion(model(x_adv),y)
    return loss_robust
