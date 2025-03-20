from __future__ import division
import warnings
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from misc.image import *
from misc.utils import *
import nni
from nni.utils import merge_parameter
from config import return_args, args
import time
from misc.ssim_loss import *
from misc.dms_ssim_loss import *
from misc.losses import *
from tqdm import tqdm
warnings.filterwarnings('ignore')

if args.del_seed:
    print("random seed is not fixed ...")
else:
    print("random seed is fixed ...")
    setup_seed(args.seed)

def cal_spatial_abstraction_loss(output, target, levels=3):
    criterion = nn.MSELoss()
    sa_loss = None
    est = output
    gt = target
    pool = nn.MaxPool2d(kernel_size=2,stride=2)
    for _ in range(levels):
        
        est = pool(est)
        gt = pool(gt)
        if sa_loss:
            sa_loss += criterion(est, gt)
        else:
            sa_loss = criterion(est, gt)
    return sa_loss
def cal_mse(est,gt):
    return torch.mean((est-gt)**2)

def cal_loss(output, target, loss="MSE"):
    criterion = nn.MSELoss(size_average=False).cuda()
    if loss == "MSE":
        loss = criterion(output,target).cuda()
    elif loss == "CALoss":
        mae_critetion = nn.L1Loss().cuda()    
        loss = criterion(output,target).cuda() + 0.001 * cal_spatial_correlation_loss(output,target).cuda() +  0.01 * mae_critetion(output,target)    

    return loss  
def cosine_similarity(stu_map, tea_map):
    similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum()
    return loss    
    
def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
        val_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
        val_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy' 
        test_file = './npydata/qnrf_test.npy'
        val_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'UCF50_1':
        train_file = './npydata/npy/ucf50_train1.npy'
        test_file = './npydata/npy/ucf50_test1.npy'
        val_file = './npydata/npy/ucf50_test1.npy'
    elif args['dataset'] == 'UCF50_2':
        train_file = './npydata/npy/ucf50_train2.npy'
        test_file = './npydata/npy/ucf50_test2.npy'
        val_file = './npydata/npy/ucf50_test2.npy'
    elif args['dataset'] == 'UCF50_3':
        train_file = './npydata/npy/ucf50_train3.npy'
        test_file = './npydata/npy/ucf50_test3.npy'
        val_file = './npydata/npy/ucf50_test3.npy'
    elif args['dataset'] == 'UCF50_4':
        train_file = './npydata/npy/ucf50_train4.npy'
        test_file = './npydata/npy/ucf50_test4.npy'
        val_file = './npydata/npy/ucf50_test4.npy'
    elif args['dataset'] == 'UCF50_5':
        train_file = './npydata/npy/ucf50_train5.npy'
        test_file = './npydata/npy/ucf50_test5.npy'
        val_file = './npydata/npy/ucf50_test5.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train_2048.npy'
        test_file = './npydata/nwpu_val_2048.npy'
    elif args['dataset'] == 'CARPK':
        train_file = './npydata/carpk_train.npy'
        test_file = './npydata/carpk_test.npy'
        val_file = './npydata/carpk_test.npy'
    elif args['dataset'] == 'PUCPR':
        train_file = './npydata/pucpr_train.npy'
        test_file = './npydata/pucpr_test.npy'
        val_file = './npydata/pucpr_test.npy'
    elif args['dataset'] == 'large':
        train_file = './npydata/large_train.npy'
        test_file = './npydata/large_val.npy'
        val_file = './npydata/large_test.npy'
    elif args['dataset'] == 'small':
        train_file = './npydata/small_train.npy'
        test_file = './npydata/small_val.npy'
        val_file = './npydata/small_test.npy'
    elif args['dataset'] == 'TRANCOS':
        train_file = './npydata/trancos_train.npy'
        test_file = './npydata/trancos_test.npy'
        val_file = './npydata/trancos_val.npy'

        
    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()
    
    if args["debug"]:
        logger.info("debug mode ...")
        train_list = train_list[:len(train_list)//10]
        test_list = test_list[:len(test_list)//10]
        args["start_val"] = 1
        args["val_freq"] = 2
    else:
        args["start_val"] = 200  
        args["val_freq"] = 5

    model = get_model(args).to('cuda')
    optimizer = torch.optim.Adam(
        [  
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    
    logger.info(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    logger.info(f"best_pred: {args['best_pred']} start_epoch: {args['start_epoch']}")

    if args['preload_data'] == True:
        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list
        
    best_epoch = 0
    for epoch in tqdm(range(args['start_epoch'], args['epochs']), desc=args["task_name"], leave=False):

        start = time.time()
        train(train_data, model, cal_loss, optimizer, epoch, args)
        end1 = time.time()

        if epoch % args["val_freq"] == 0 and epoch >= args["start_val"] or epoch == (args["epochs"] -1):
            prec1, mse, visi = validate(test_data, model, args, epoch)

            end2 = time.time()

            is_best = prec1 < args['best_pred']
            if is_best:
                args["best_mse"] = mse
                best_epoch = epoch
            args['best_pred'] = min(prec1, args['best_pred'])

            logger.info(
                f"best_epoch {best_epoch} * best MAE {args['best_pred']:.3f} * best MSE {args['best_mse']:.3f} {args['save_path']} \
                    {end1 - start} {end2 - end1}"
            )     
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, visi, is_best, args['save_path'])
    logger.info("start test best_model ...")
    checkpoint = torch.load(f"{args['save_path']}/model_best.pth")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    args["save_path"] = args["save_path"].replace("train", "test")
    logger.info(args["save_path"])
    os.makedirs(args["save_path"], exist_ok=True)
     
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    val_data = pre_data(val_list, args, train=False)
    
    _, _, visi = validate(val_data, model, args, epoch+1, "test")
    vis_test(visi, args)

def vis_test(visi, args):
    for i in range(len(visi)):
        img = visi[i][0]
        output = visi[i][1]
        target = visi[i][2]
        fname = visi[i][3]
        save_results(img, target, output, str(args["save_path"]), fname[0])

def pre_data(train_list, args, train):
    logger.info("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:
            continue
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys

def train(Pre_data, model, cal_loss, optimizer, epoch, args):    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    logger.info('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))
    
    model.train()
    end = time.time()
    for i, (fname, img, fidt_map, kpoint) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()
        if args["network"] == "os_kd" or args["network"] == "student_kd":
            x1, x2, x3, x4, x5, s_out, experts = model(img)
            d6 = s_out
            loss1 = cosine_similarity(experts[0], x1)
            loss2 = cosine_similarity(experts[1], x2)
            loss3 = cosine_similarity(experts[2], x3)
            loss4 = cosine_similarity(experts[3], x4)
            loss5 = cosine_similarity(experts[4], x5)
            loss6 = cal_loss(d6, fidt_map, args["loss"])
            loss = loss1 + loss2 + loss3  + loss4  + loss5 + loss6  
        else:    
            d6 = model(img)
            loss = cal_loss(d6, fidt_map, args["loss"])
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % (len(train_loader) // 10 + 1) == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    if epoch > 10:    
        clearml_logger.report_scalar("loss", "loss", losses.avg, epoch)
        if args["network"] == "os_kd" or args["network"] == "student_kd":
            clearml_logger.report_scalar("loss1", "loss1", loss1, epoch)
            clearml_logger.report_scalar("loss2", "loss2", loss2, epoch)
            clearml_logger.report_scalar("loss3", "loss3", loss3, epoch)
            clearml_logger.report_scalar("loss4", "loss4", loss4, epoch)
            clearml_logger.report_scalar("loss5", "loss5", loss5, epoch)
            clearml_logger.report_scalar("loss6", "loss6", loss6, epoch)
            
def validate(Pre_data, model, args, epoch, mode="val"):
    logger.info('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()
    mae = 0.0
    mse = 0.0
    visi = []
    index = 0
    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')
    f_loc = open("./local_eval/A_localization.txt", "w+")
    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
        count = 0
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            if args["network"] == "v":         
                if img.size(-1) == 256:
                    img = img.permute(0,1,3,2) 
                    fidt_map = fidt_map.permute(0,1,3,2)                  
                raw_h, raw_w = img.shape[2:] 
                patches, _ = sliding_window(img,stride=128)
                patches = torch.from_numpy(patches).float().cuda()
                d6 = model(patches)
                d6 = window_composite(d6, stride=128)
                d6 = d6[:, :, :raw_w]
            elif args["network"] == "os_kd" or args["network"] == "student_kd":
                x1, x2, x3, x4, x5, s_out, experts = model(img)
                d6 = s_out
            else:
                d6 = model(img)
            count, pred_kpoint, f_loc = LMDS_counting(d6, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(d6.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

        gt_count = torch.sum(kpoint).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        if mode == "val":
            if i % (len(test_loader) // 10 + 1) == 0:
                logger.info('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
                visi.append(
                    [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                    fname])
                index += 1
        elif mode == "test":
            if i % 1 == 0:
                logger.info('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
                visi.append(
                    [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                    fname])
                index += 1
            

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    nni.report_intermediate_result(mae)
    logger.info(f"\n* MAE {mae:.3f}\n* MSE {mse:.3f}")
    for key, value in {"mae": mae, "mse": mse}.items():
        clearml_logger.report_scalar(key, key, value, epoch)
    return mae, mse, visi

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    if args['dataset'] == 'UCF_QNRF' :
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
        '/home/dkliang/projects/synchronous/datasets/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
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

def sliding_window(image, window_size = (256, 256), stride = 128):
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    image = np.pad(image, ((0, 0), (0, stride - image.shape[1] % stride), (0, 0)), 'constant')
    h, w, _ = image.shape
    assert h == 256, "FSC-147 assume image height is 384." 
    patches = []
    intervals = []
    for i in range(0, w - window_size[1] + 1, stride):
        patch = image[:, i:i + window_size[1], :]
        patches.append(patch)
        intervals.append([i, i + window_size[1]])
    return np.array(patches).transpose(0,3,1,2), np.array(intervals)


def window_composite(patches, window_size = (256, 256), stride = 128):  

    image = None
    patch_h, patch_w = window_size
    for i, patch in enumerate(patches):
        if i == 0:
            image = patch 

        else:
            blend_width = patch_w - stride
            prev_to_blend = image[:, :, -blend_width:]
            next_to_blend = patch[:, :, :blend_width]
            blend_factor = torch.sigmoid(torch.tensor(np.linspace(-3, 3, blend_width))).to(image.device)
            blend = (1-blend_factor) * prev_to_blend + blend_factor * next_to_blend
            image[:, :, -blend_width:] = blend
            patch_remain = patch[:, :, blend_width:]
            image = torch.cat([image, patch_remain], dim = -1)
    return image

if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))
    os.makedirs(params["save_path"], exist_ok=True)  
    params["task_name"] = "_".join(params["save_path"].split("/")[-2:])  
    global clearml_logger       
    clearml_logger = set_clearml("vehicle counting", task_name="Moe_student_CARPK")   
    global logger 
    logger = get_logger("Train", f"{params['save_path']}/train.log")
    
    params["save_path"] = os.path.join(params["save_path"], "train") 
    os.makedirs(params["save_path"], exist_ok=True)
    logger.debug(tuner_params)
    logger.info(params)
    main(params)
