import numpy as np
import torch
import glog as log
from config import IN_CHANNELS, IMAGE_SIZE
from square_attack_RL.training.environment import Environment
from square_attack_RL.training.state import State


def normalize(t):
    """
    Return the norm of a tensor (or numpy) along all the dimensions except the first one
    :param t:
    :return:
    """
    _shape = t.shape
    batch_size = _shape[0]
    num_dims = len(_shape[1:])
    if torch.is_tensor(t):
        norm_t = torch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
        norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
        return norm_t
    else:
        _norm = np.linalg.norm(
            t.reshape([batch_size, -1]), axis=1
        ).reshape([batch_size] + [1] * num_dims)
        return _norm + (_norm == 0) * np.finfo(np.float64).eps

def l2_proj_maker(xs, eps):
    orig = xs.clone()

    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (normalize(delta) > eps).float()
        x = (orig + eps * delta / normalize(delta)) * out_of_bounds_mask
        x += new_x * (1 - out_of_bounds_mask)
        return x
    return proj

def linf_proj_maker(xs, eps):
    """
    makes an linf projection function such that new points
    are projected within the eps linf-balls centered around xs
    :param xs:
    :param eps:
    :return:
    """
    if torch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            return orig_xs + torch.clamp(new_xs - orig_xs, - eps, eps)
    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            return np.clip(new_xs, orig_xs - eps, orig_xs + eps)
    return proj

def proj_replace(orig_img, xs, suggest_xs, norm, eps, done_mask):
    if norm == 'l2':
        l2_proj = l2_proj_maker(orig_img, eps)
        _proj = lambda xx: torch.clamp(l2_proj(xx), 0, 1)
    elif norm == 'linf':
        linf_proj = linf_proj_maker(orig_img, eps)
        _proj = lambda xx: torch.clamp(linf_proj(xx), 0, 1)

    suggest_xs = _proj(suggest_xs) # _proj函数在run函数开始处定义
    # replace xs only if not done
    xs = suggest_xs * (1. - done_mask) + xs * done_mask
    return xs


def test(data_loader, agent, model, episode, args):
    agent.model.eval()
    sum_reward = 0
    current_state = State((args.batch_size, IN_CHANNELS[args.dataset], IMAGE_SIZE[args.dataset][0],
                            IMAGE_SIZE[args.dataset][1]), args.norm, args.epsilon)
    test_data_size = len(data_loader)
    not_done_all = torch.zeros(args.batch_size * 100).float()
    correct_all = torch.zeros_like(not_done_all)
    for idx, (raw_x, label) in enumerate(data_loader):
        if idx >= 100:
            break
        environment = Environment()
        raw_x = raw_x.cuda()
        label = label.cuda()
        batch_size = raw_x.size(0)
        assert batch_size == args.batch_size
        selected = torch.arange(idx * batch_size,
                                min((idx + 1) * batch_size, len(data_loader.dataset)))
        with torch.no_grad():
            logit = model(raw_x)
        pred = logit.argmax(dim=1)
        correct = pred.eq(label).float()
        correct_all[selected] = correct.detach().cpu()
        not_done = correct.clone()
        orig_img = raw_x.clone()
        xs = raw_x
        current_state.reset(xs)
        environment.get_reward(model, current_state.image, label)
        num_axes = len(raw_x.shape[1:])
        for t in range(0, 100):
            action = agent.act(current_state.image)
            sugg_xs = current_state.step(action)
            xs = proj_replace(orig_img, xs, sugg_xs, args.norm, args.epsilon, 1.0 - not_done.view(-1, *[1] * num_axes))
            with torch.no_grad():
                adv_logit = model(xs)
            adv_pred = adv_logit.argmax(dim=1)
            not_done = not_done * adv_pred.eq(label).float()
            reward = environment.get_reward(model, current_state.image, label)
            sum_reward += reward.mean().item() * np.power(args.gamma, t)
        not_done_all[selected] = not_done.cpu().detach()
        agent.stop_episode()
    attack_success_rate = 1 - not_done_all[correct_all.byte()].mean().item()
    log.info("[TEST] {e}-th episode, attack success rate: {a}, total reward {r}".format(e=episode,
                                                                      r=sum_reward / test_data_size, a=attack_success_rate))
    agent.model.train()
