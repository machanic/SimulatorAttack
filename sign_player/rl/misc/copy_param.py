
def copy_param(target_model, source_model):
    """Copy parameters of a link to another link."""
    target_model.load_state_dict(source_model.state_dict())


def copy_grad(target_model, source_model):
    for net1, net2 in zip(source_model.named_parameters(), target_model.named_parameters()):
        if net1[1].grad is None:
            print(net1[0] + " None!")
        net2[1].grad = net1[1].grad.clone()
