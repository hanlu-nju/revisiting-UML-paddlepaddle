import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader

from model.dataloader.samplers import CategoriesSampler
from model.models import wrappers
from model.models.protonet import ProtoNet
from model.models.tsp_head import TSPHead
from model.utils import get_dataset
import paddle.fluid as fluid


def examplar_collate(batch):
    X, Y = [], []
    for b in batch:
        X.append(paddle.stack(b[0]))
        Y.append(b[1])
    X = paddle.stack(X)
    label = paddle.to_tensor(Y)
    img = paddle.concat(tuple(paddle.transpose(X, (1, 0, 2, 3, 4))), axis=0)
    # (repeat * class , *dim)
    return img, label


def get_dataloader(args):
    num_device = fluid.cuda_places()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = args.num_workers * num_device if args.multi_gpu else args.num_workers

    trainset = get_dataset(args.dataset, 'train', args.unsupervised, args, augment=args.augment)

    args.num_classes = min(len(trainset.wnids), args.num_classes)

    if args.unsupervised:
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=examplar_collate,
                                  drop_last=True)
    else:
        train_sampler = CategoriesSampler(trainset.label,
                                          num_episodes,
                                          max(args.way, args.num_classes),
                                          args.shot + args.query)

        train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  )

    valset = get_dataset(args.dataset, 'val', args.unsupervised, args)
    testsets = dict(((n, get_dataset(n, 'test', args.unsupervised, args)) for n in args.eval_dataset.split(',')))
    args.image_shape = trainset.image_shape
    return train_loader, valset, testsets


def prepare_model(args):
    args.device = device = 'gpu' if len(fluid.cuda_places()) > 0 else 'cpu'
    paddle.device.set_device(args.device)
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        # if args.augment == 'moco':
        #     pretrained_dict = torch.load(args.init_weights)['state_dict']
        #     pretrained_dict = {'encoder' + k[len('encoder_q'):]: v for k, v in pretrained_dict.items() if
        #                        k.startswith('encoder_q')}
        # else:
        try:
            pretrained_dict = paddle.load(args.init_weights, map_location=args.device)
        except:
            import pickle

            with open(args.init_weights, 'rb') as fp:
                pretrained_dict = pickle.load(fp)
        keys = ['params', 'state_dict']
        for k in keys:
            if k in pretrained_dict:
                pretrained_dict = pretrained_dict[k]
                break
        # pretrained_dict = torch.load(args.init_weights)['params']
        # if args.backbone_class == 'ConvNet':
        #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.additional != 'none':
        model = getattr(wrappers, args.additional + 'Wrapper')(args, model)
        # model = TaskContrastiveWrapper(args, model)

    # if len(fluid.cuda_places()) > 0:
    #     paddle.backends.cudnn.benchmark = True

    model = model.to(device)
    # if args.multi_gpu:
    #     model.encoder = nn.DataParallel(model.encoder, dim=0)
    #     para_model = model.to(device)
    # else:
    para_model = model.to(device)
    # if args.finetune:
    #     model.eval()
    #     para_model.eval()
    # print(model.state_dict().keys())
    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    print('top params', [k for k, v in model.named_parameters() if 'encoder' not in k])
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    param_groups = [{'params': list(model.encoder.parameters())},
                    {'params': top_para, 'lr': args.lr * args.lr_mul}]
    # param_groups = model.parameters()
    # param = dict(model.named_parameters())
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr.StepDecay(
            learning_rate=args.lr,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr.MultiStepDecay(
            learning_rate=args.lr,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr.CosineAnnealingDecay(
            T_max=args.max_epoch + 1,
            learning_rate=args.lr,
            eta_min=0  # a tuning parameter
        )
    elif args.lr_scheduler == 'constant':
        lr_scheduler = optim.lr.LambdaDecay(
            learning_rate=args.lr,
            lr_lambda=lambda ep: 1  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    if args.backbone_class in ['ConvNet']:
        optimizer = optim.Adam(
            parameters=param_groups,
            learning_rate=lr_scheduler,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    else:
        optimizer = optim.Momentum(parameters=param_groups,
                                   learning_rate=lr_scheduler,
                                   momentum=args.mom,
                                   use_nesterov=True,
                                   weight_decay=args.weight_decay
                                   )

    return optimizer, lr_scheduler
