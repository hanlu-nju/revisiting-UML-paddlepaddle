import paddle
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import pprint
from model.utils import set_gpu
from model.utils import get_command_line_parser
from model.utils import postprocess_args
if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    args.eval = True
    args = postprocess_args(args, train=False)
    pprint(vars(args))
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)
    trainer.evaluate_model(path=args.path)
    print(args.save_path)
