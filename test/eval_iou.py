import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes, ctem

from lednet.py import Net


from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    ToLabel(),
])


def reset_cuda_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def print_cuda_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[CUDA Mem] {tag} | allocated={alloc:.1f} MB, reserved={reserved:.1f} MB, peak={peak:.1f} MB")

def main(args):
    global NUM_CLASSES
    if args.dataset == 'ctem':
        NUM_CLASSES = 6
    else:
        NUM_CLASSES = 20

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    input_transform = Compose([
        Resize((args.height, args.width), Image.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize((args.height, args.width), Image.NEAREST),
        ToLabel(),
    ])

    if args.dataset == 'ctem':
        loader = DataLoader(ctem(args.datadir, input_transform, target_transform, subset=args.subset),
                            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    else:
        loader = DataLoader(cityscapes(args.datadir, input_transform, target_transform, subset=args.subset),
                            num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=-1, ignore_label_value=args.ignore_label)

    start = time.time()
    if args.cuda_mem:
        reset_cuda_mem()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        if args.dataset == 'ctem':
            filenameSave = os.path.basename(filename[0])
        else:
            filenameSave = filename[0].split("leftImg8bit/")[1]

        print(step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    if args.dataset == 'ctem':
        for idx, iou_str in enumerate(iou_classes_str):
            print(iou_str, f"class_{idx}")
    else:
        print(iou_classes_str[0], "Road")
        print(iou_classes_str[1], "sidewalk")
        print(iou_classes_str[2], "building")
        print(iou_classes_str[3], "wall")
        print(iou_classes_str[4], "fence")
        print(iou_classes_str[5], "pole")
        print(iou_classes_str[6], "traffic light")
        print(iou_classes_str[7], "traffic sign")
        print(iou_classes_str[8], "vegetation")
        print(iou_classes_str[9], "terrain")
        print(iou_classes_str[10], "sky")
        print(iou_classes_str[11], "person")
        print(iou_classes_str[12], "rider")
        print(iou_classes_str[13], "car")
        print(iou_classes_str[14], "truck")
        print(iou_classes_str[15], "bus")
        print(iou_classes_str[16], "train")
        print(iou_classes_str[17], "motorcycle")
        print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")
    if args.cuda_mem:
        print_cuda_mem(tag=f"eval-{args.subset}")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="lednet_trained.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dataset', default='cityscapes', choices=['cityscapes', 'ctem'])
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--cuda-mem', action='store_true')

    args = parser.parse_args()

    if args.dataset == 'ctem':
        args.height = 540
        args.width = 960
        args.ignore_label = 255

    main(args)
