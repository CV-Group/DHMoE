import math
import os
import warnings

import cv2
import nni
import numpy as np
import scipy
import torch
import torch.nn as nn
from nni.utils import merge_parameter
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import dataset_nwpu
from config import args, return_args
from misc.image import load_data_fidt
from misc.utils import get_logger, setup_seed
from Networks.MOE_KD import Model as os_kd

warnings.filterwarnings('ignore')

setup_seed(args.seed)

def main(args):
    if args['dataset'] == 'ShanghaiA':
        test_file = './npydata_jy/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        test_file = './npydata_jy/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        test_file = './npydata_jy/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        test_file = './npydata_jy/jhu_test.npy'
    elif args['dataset'] == 'NWPU':
        test_file = './npydata/nwpu_test_2048.npy'
    elif args['dataset'] == 'UCF_50':
        test_file = './npydata/ucf50_test4.npy'

    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    model = os_kd()
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ])

    logger.info(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    logger.info(f"{args['best_pred']} {args['start_epoch']}")


    test_data = pre_data(val_list)

    '''inference '''
    visi = validate(test_data, model, args)

    logger.info(f"\nThe visualizations are provided in {args['save_path']}")
    
    for i in range(len(visi)):
        output = visi[i][1]
        fname = visi[i][2]
        save_results(output, str(args['save_path']), fname[0])
        

def save_results(density_map, output_dir, fname="results.png"):
    density_map[density_map < 0] = 0

    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    density_map = density_map.astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, 2)
    
    cv2.imwrite(
        os.path.join(".", output_dir, fname).replace(".jpg", ".jpg"), density_map
    )

     
def pre_data(train_list):
    logger.info("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img = Image.open(Img_path).convert('RGB')

        blob = {}
        blob['img'] = img
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys


def validate(Pre_data, model, args):
    logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset_nwpu.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1)
    model.eval()
    visi = []
    index = 0
    if not os.path.exists('./local_eval/point_files'):
        os.makedirs('./local_eval/point_files')
    f_loc = open("./local_eval/point_files/A_localization.txt", "w+")

    for i, (fname, img) in enumerate(
        tqdm(test_loader, desc=f"{args['dataset']}", leave=False)):

        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            x1, x2, x3, x4, x5, s_out, experts = model(img)
            d6 = s_out
            count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

        if i % 1 == 0:
            logger.info('{fname} Pred {pred:.4f}'.format(fname=fname[0], pred=count))
            visi.append(
                [img.data.cpu().numpy(), d6.data.cpu().numpy(),fname])
            index += 1

    return visi


def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    if args['dataset'] == 'UCF_QNRF':
        input = nn.functional.avg_pool2d(input, (3, 3), stride=1, padding=1)
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc


def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map


def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        '/home/dataset/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data


def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    params["save_path"] = "/scratch/nwpu"
    os.makedirs(params["save_path"], exist_ok=True)
    global logger
    logger = get_logger("Test", "/scratch/nwpu/test.log")
    logger.debug(tuner_params)
    logger.info(params)
    main(params)