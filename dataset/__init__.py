from .cityscapes import CSTrainValSet
from .ade20k import ADETrainSet, ADEDataValSet
from .coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet


def get_dataset(args):
    if args.dataset == 'citys':
        train_dataset = CSTrainValSet(args.data,
                                      list_path='./dataset/list/cityscapes/train.lst',
                                      ignore_label=args.ignore_label,
                                      max_iters=args.max_iterations*args.batch_size,
                                      crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = CSTrainValSet(args.data,
                                    list_path='./dataset/list/cityscapes/val.lst',
                                    ignore_label=args.ignore_label,
                                    crop_size=(1024, 2048), scale=False, mirror=False)
    elif args.dataset == 'ade20k':
        train_dataset = ADETrainSet(args.data,
                                    max_iters=args.max_iterations*args.batch_size,
                                    ignore_label=args.ignore_label,
                                    crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = ADEDataValSet(args.data)
    elif args.dataset == 'coco_stuff_164k':
        train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt',
                                              max_iters=args.max_iterations*args.batch_size,
                                              ignore_label=args.ignore_label,
                                              crop_size=args.crop_size, scale=True, mirror=True)
        val_dataset = CocoStuff164kValSet(
            args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
    else:
        raise ValueError('dataset unfind')

    return train_dataset, val_dataset
