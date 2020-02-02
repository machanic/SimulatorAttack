import argparse
import random
import numpy as np

def get_args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10", "imagenet"], default="mnist")
    parser.add_argument("-s", "--save", default="./saved_results")
    parser.add_argument("-a", "--attack", choices=["white", "black"], default="white")
    parser.add_argument("-n", "--numimg", type=int, default=0, help="number of test images to attack")
    parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to use default value")
    parser.add_argument("-p", "--print_every", type=int, default=100, help="print objs every PRINT_EVERY iterations")
    parser.add_argument("-o", "--early_stop_iters", type=int, default=100,
                        help="print objs every EARLY_STOP_ITER iterations, 0 is maxiter//10")
    parser.add_argument("-f", "--firstimg", type=int, default=0)
    parser.add_argument("-b", "--binary_steps", type=int, default=0)
    parser.add_argument("-c", "--init_const", type=float, default=0.0)
    parser.add_argument("-z", "--use_zvalue", action='store_true')
    parser.add_argument("-u", "--untargeted", action='store_true')
    parser.add_argument("-r", "--reset_adam", action='store_true', help="reset adam after an initial solution is found")
    parser.add_argument("--use_resize", action='store_true', help="resize image (only works on imagenet!)")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=1216)
    parser.add_argument("--solver", choices=["adam", "newton", "adam_newton", "fake_zero"], default="adam")
    parser.add_argument("--save_ckpts", default="", help="path to save checkpoint file")
    parser.add_argument("--load_ckpt", default="", help="path to numpy checkpoint file")
    parser.add_argument("--start_iter", default=0, type=int,
                        help="iteration number for start, useful when loading a checkpoint")
    parser.add_argument("--init_size", default=32, type=int, help="starting with this size when --use_resize")
    parser.add_argument("--uniform", action='store_true', help="disable importance sampling")
    args = vars(parser.parse_args())
    args['lr'] = 1e-2
    args['inception'] = False
    args['use_tanh'] = True
    # args['use_resize'] = False
    if args['maxiter'] == 0:
        if args['attack'] == "white":
            args['maxiter'] = 1000
        else:
            if args['dataset'] == "imagenet":
                if args['untargeted']:
                    args['maxiter'] = 1500
                else:
                    args['maxiter'] = 50000
            elif args['dataset'] == "mnist":
                args['maxiter'] = 3000
            else:
                args['maxiter'] = 1000
    if args['init_const'] == 0.0:
        if args['binary_steps'] != 0:
            args['init_const'] = 0.01
        else:
            args['init_const'] = 0.5
    if args['binary_steps'] == 0:
        args['binary_steps'] = 1
    # set up some parameters based on datasets
    if args['dataset'] == "imagenet":
        args['inception'] = True
        args['lr'] = 2e-3
        # args['use_resize'] = True
        # args['save_ckpts'] = True
    # for mnist, using tanh causes gradient to vanish
    if args['dataset'] == "mnist":
        args['use_tanh'] = False
    # when init_const is not specified, use a reasonable default
    if args['init_const'] == 0.0:
        if args['binary_search']:
            args['init_const'] = 0.01
        else:
            args['init_const'] = 0.5

    print('Done...')
    print('Using', args['numimg'], 'test images')
    # setup random seed
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    

