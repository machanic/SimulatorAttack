import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import argparse
import json
import os
from types import SimpleNamespace
import glog as log

from TREMBA.FCN import *
from TREMBA.FCN import ImagenetDecoder, ImagenetEncoder
from TREMBA.utils import MarginLoss
from config import MODELS_TRAIN_STANDARD, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.standard_model import StandardModel


def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--config', default='/home1/machen/meta_perturbations_black_box_attack/configures/TREMBA_attack.json', help='config file')
    parser.add_argument('--targeted',action="store_true")
    parser.add_argument('--dataset',required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    targeted_str = "untargeted" if not args.targeted else "targeted"
    if args.targeted:
        save_name = "train_pytorch_model/TREMBA/{}_{}_generator.pth.tar".format(args.dataset, targeted_str)
    else:
        save_name = "train_pytorch_model/TREMBA/{}_{}_generator.pth.tar".format(args.dataset, targeted_str)
    set_log_file(os.path.dirname(save_name) + "/train_{}_{}.log".format(args.dataset, targeted_str))
    with open(args.config) as config_file:
        state = json.load(config_file)["train"][targeted_str]
        state = SimpleNamespace(**state)
        state.targeted = args.targeted
        state.dataset = args.dataset
        state.batch_size = args.batch_size
    device = torch.device(args.gpu)
    train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, state.batch_size, True)
    val_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, state.batch_size, False)
    nets = []
    log.info("Initialize pretrained models.")
    for model_name in MODELS_TRAIN_STANDARD[args.dataset]:
        pretrained_model = StandardModel(args.dataset, model_name, no_grad=False)
        # pretrained_model.cuda()
        pretrained_model.eval()
        nets.append(pretrained_model)
    log.info("Initialize over!")
    model = nn.Sequential(
        ImagenetEncoder(),
        ImagenetDecoder(args.dataset)
    )
    model = model.cuda()
    optimizer_G = torch.optim.SGD(model.parameters(), state.learning_rate_G, momentum=state.momentum,
                                  weight_decay=0, nesterov=True)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=state.epochs // state.schedule,
                                                  gamma=state.gamma)
    hingeloss = MarginLoss(margin=state.margin, target=state.targeted)

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    log.info("After training for {} epochs, model will be saved to {}".format(state.epochs, save_name))
    def train():
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            nat = data.cuda()
            if state.targeted:
                label = state.target_class
            else:
                label = label.cuda()
            optimizer_G.zero_grad()
            for net in nets:
                net.cuda()
                noise = model(nat)
                adv = torch.clamp(noise * state.epsilon + nat, 0, 1)
                logits_adv = net(adv)
                loss_g = hingeloss(logits_adv, label)
                loss_g.backward()
                net.cpu()
            optimizer_G.step()

    def test():
        model.eval()
        loss_avg = [0.0 for i in range(len(nets))]
        success = [0 for i in range(len(nets))]
        for batch_idx, (data, label) in enumerate(val_loader):
            nat = data.cuda()
            if state.targeted:
                label = state.target_class
            else:
                label = label.cuda()
            noise = model(nat)
            adv = torch.clamp(noise * state.epsilon + nat, 0, 1)
            for j in range(len(nets)):
                net = nets[j]
                net.cuda()
                logits = net(adv)
                loss = hingeloss(logits, label)
                loss_avg[j] += loss.item()
                net.cpu()
                if state.targeted:
                    success[j] += int((torch.argmax(logits, dim=1).eq(label)).sum())
                else:
                    success[j] += int((torch.argmax(logits, dim=1).eq(label)).sum())
        state.test_loss = [loss_avg[i] / len(val_loader) for i in range(len(loss_avg))]
        state.test_successes = [success[i] / len(val_loader.dataset) for i in range(len(success))]
        state.test_success = 0.0
        for i in range(len(state.test_successes)):
            state.test_success += state.test_successes[i] / len(state.test_successes)

    for epoch in range(state.epochs):
        scheduler_G.step(epoch)
        state.epoch = epoch
        train()
        log.info("Train over {}-th epoch".format(epoch))
        torch.cuda.empty_cache()
        if epoch % 100 == 0:
            with torch.no_grad():
                test()
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, save_name)
        log.info("Epoch {}, Current success: {}".format(epoch, state.test_success))
