import os
import cv2
import gc
import random
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import argparse
from glob import glob

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
from models.OEFT import OEFT

parser = argparse.ArgumentParser(description='Code to optimize')
parser.add_argument('--device',                  help='cuda | cuda:0 | cpu',                   default="cuda", type=str)
parser.add_argument('--device_num',              help='which GPUs to use',                     default="0",  type=str)
parser.add_argument('--sample_freq',        help="sampling frequency of saving results",  default=500, type=float)

""" optimizer.py setting """
parser.add_argument('--content_root', help='folder of content images',             default="../OEFT/example_210905/input/", type=str)
parser.add_argument('--style_root',   help='folder of style images',               default="../OEFT/example_210905/style/", type=str)
parser.add_argument('--save_root',    help='folder of saving results',             default="../OEFT/example_210929/experiment", type=str)
parser.add_argument('--fileType',    help='png|jpg',             default="png", type=str)

parser.add_argument('--keys',         help='vgg layer names',                      default=['r12', 'r22', 'r34', 'r44', 'r54'], nargs="+")
parser.add_argument('--iter',         help="number of iteration for optimization", default=1000, type=int)
parser.add_argument('--img_size',     help="size of input image",                  default=256, type=int)

parser.add_argument('--pretrained',     help="use pre-trained network or not",     action="store_false")
parser.add_argument('--denorm',       help="size of input image",                  action="store_false")

parser.add_argument('--lr',           help="learning rate",                        default=1e-4, type=float)
parser.add_argument('--beta1',        help="optimizer parameter",                  default=0.5, type=float)
parser.add_argument('--beta2',        help="optimizer parameter",                  default=0.999, type=float)
parser.add_argument('--weight_decay', help="weight_decay",                         default=1e-4, type=float)

""" OEPT.py setting """
parser.add_argument('--warpFeat',     help="use warped feature as decoder input",                                  action="store_true")
parser.add_argument('--warpMv',       help="use warped feature as moving averaged feature with content feature",   action="store_true")
parser.add_argument('--warpRes',      help="use warped image as residual",                                         action="store_true")
parser.add_argument('--cycle',        help="use cycle consistency regularization",                                 action="store_true")

parser.add_argument('--res_wt',       help="weight between decoder output and residual warped img",                default=8/9, type=float)
parser.add_argument('--cycle_wt',     help="weight of cycle consistency regularization",                           default=1.,  type=float)

# 256, 128, 64, 32, 16
parser.add_argument('--nce_wt',       help='nce loss weights from each layer[256-16]',                             default=[1/8*1/4,  1/4*1/4, 1/2*1/4, 1.*1/4, 1.*1/4], nargs="+")
parser.add_argument('--nns_wt',       help='NN style loss weights from each layer[256-16]',                        default=[1/16*1/4, 1/8*1/4, 1/4*1/4, 1/2*1/4, 1.*1/4], nargs="+")

parser.add_argument('--nce_temp',     help="temperature for nce",                                                  default=0.07, type=float)
parser.add_argument('--nns_temp',     help="temperature for nns",                                                  default=0.05, type=float)

parser.add_argument('--content_style_wt',        help="weight of between content and style",                       default=4/5, type=float)

""" corrnet.py setting """
parser.add_argument('--corr_temp',    help="temperature of correlation module",                                    default=0.01, type=float)
parser.add_argument('--mv_wt',        help="weight of moving average",                                             default=0.6, type=float)

mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
std =  np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
GPU_list = args.device_num
len_GPU = len( GPU_list.split(","))
print("@@@@@@@@@@@@ len_GPU: ", len_GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list

#  Load content and style
print(os.listdir('./'))
def img_load(path):
    img = cv2.imread(path)[::,::,::-1] # BGR to RGB, [0-255]
    return img

def toPIL(img):
    # image range should be [0-255] for converting.
    img_type = str(type(img))
    if 'numpy' in img_type:
        img = Image.fromarray(img)
    elif 'torch' in img_type:
        img = transforms.ToPILImage()(img).convert("RGB")
    return img

if __name__ == "__main__":
    # parse options
    keys = args.keys
    content_root =   args.content_root
    style_root   =   args.style_root

    if not os.path.exists(args.content_root):
        print("!!! args.content_root does not exist !!!")
        exit()
    if not os.path.exists(args.style_root):
        print("!!! args.style_root does not exist !!!")
        exit()

    content_list = glob( os.path.join(content_root, "*.{}".format(args.fileType)) )
    style_list = glob( os.path.join(style_root, "*.{}".format(args.fileType) ) )

    if len(content_list) < 1 or len(style_list) < 1:
        print("!!! The number of content and style images should be more than 1 !!!")
        exit()
    content_list.sort()
    style_list.sort()

    print("@@@@@@@@@ len(content_list): ", len(content_list))
    for z in range(len(content_list)):

        random_seed = 1006
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

        """ start iteration """
        torch.cuda.empty_cache() # remove all caches
        try:
            """ content, style image path indexing """
            content_path = content_list[z]   # './examples/input/in11.png'
            style_path   = style_list[z]     # './examples/style/tar11.png'
        
            """ img load """
            content = img_load(content_path)
            style  = img_load(style_path)
            content_256 = content.copy()
            style_256 = style.copy()
        except Exception as e:
            print("image loading error : ", e)
            continue
        
        """ Convert numpy array to PIL.Image format """
        """ and modify the range (0-255) to [0-1] """
        content = toPIL(content)
        style = toPIL(style)
        
        """ Make transform """
        transform_list = []
        img_size = (args.img_size, args.img_size)
        transform_list.append(transforms.Resize(img_size, interpolation=2)) # @@@@ args.interpol-method = 2
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize( (0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225))]
        transform = transforms.Compose(transform_list)
        
        """ do transform """
        content = transform(content)
        style = transform(style)
        content = torch.unsqueeze( content, dim=0 )
        style   = torch.unsqueeze( style, dim=0 )

        """ Load model """
        model = OEFT(args=args, pretrained=args.pretrained)
        model   = model.to(args.device)

        """ Define optimizer """
        e_optimizer = torch.optim.Adam(model.corrNet.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2) 
                                      )
        g_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2) 
                                      )

        for param_group in e_optimizer.param_groups:
            param_group['lr'] = 1e-4 #hparams.initial_learning_rate
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = 1e-4 #hparams.initial_learning_rate

        if "cuda" in args.device:
            content = content.type(torch.cuda.FloatTensor).to(args.device).detach()
            style   = style.type(torch.cuda.FloatTensor).to(args.device).detach()
            new_input = content.clone().type(torch.cuda.FloatTensor).to(args.device)
        else:
            content = content.type(torch.FloatTensor).to(args.device).detach()
            style   = style.type(torch.FloatTensor).to(args.device).detach()
            new_input = content.clone().type(torch.FloatTensor).to(args.device)

        warp_result = None
        dec_result = None
        cycle_result = None
        count = 0

        model.train()
        start = time.time()
        prog_bar = tqdm(range(args.iter))
        for i in prog_bar:
            if count == 0:
                warped_s2c_feat, warped_s2c_imgs, tr_s2c_img_decs, loss, loss_dict, nce_dict, nns_dict = model(style, content, step=i)
            else:
                if args.warpMv:
                    warped_s2c_feat = warped_s2c_feat
                    for key in args.keys:
                        warped_s2c_feat[key] = warped_s2c_feat[key].detach()
                    warped_s2c_feat, warped_s2c_imgs, tr_s2c_img_decs, loss, loss_dict, nce_dict, nns_dict = model(style, content, warped_s2c_feat, step=i)
                else:
                    warped_s2c_feat, warped_s2c_imgs, tr_s2c_img_decs, loss, loss_dict, nce_dict, nns_dict = model(style, content, step=i)

            # summary_writer.add_scalars( "Total NCE and NNS" , loss_dict, i)
            # summary_writer.add_scalars( "NCEs" , nce_dict, i)
            # summary_writer.add_scalars( "NNSs" , nns_dict, i)
            # nce_dict.update(nns_dict)
            # summary_writer.add_scalars( "NCEs and NNSs" , nce_dict, i)

            prog_bar.set_description("Pair:{}, iter:{}, loss_style:{}, loss_cont:{}, loss_cycle:{}".format(
                                                                                        z+1,
                                                                                        i+1,
                                                                                        loss_dict["L_style"], 
                                                                                        loss_dict["L_content"], 
                                                                                        loss_dict["L_cycle"])
                        )
            e_optimizer.zero_grad()
            g_optimizer.zero_grad()
            loss.backward()
            e_optimizer.step()
            g_optimizer.step()
            
            count += 1

            """ generation result """
            dec_result = tr_s2c_img_decs.clone().detach()

            """ save the results """
            if (i + 1) % args.sample_freq == 0 or i == 0 or i == args.iter-1:
                c_img = os.path.basename(content_path) # 'in11.png'
                s_img = os.path.basename(style_path)   # 'tar11.png'
                c_name = c_img.split('.')[0] # in11
                s_name = s_img.split('.')[0] # tar11

                pair_dir = '{}_'.format(z) + c_name + '_and_' + s_name
                pair_iter_dir = '{}_'.format(z) + "iter" + str(i) + "_" +c_name + '_and_' + s_name

                """ making folder to save results """
                root_iter_path = os.path.join(args.save_root, pair_dir)
                save_dir_path = os.path.join(root_iter_path,  pair_iter_dir)
                if not os.path.isdir(args.save_root):
                    os.makedirs(args.save_root)
                if not os.path.isdir(save_dir_path):
                    os.makedirs(save_dir_path)

                """ denormalization """
                if args.denorm == True:
                    result = ( np.clip(( dec_result[0].permute(1,2,0).clone().detach().cpu().numpy()) *std + mean, 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
                else:
                    result = ( np.clip(( dec_result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]

                """ change the form """
                content_save  =  (cv2.resize(content_256, (256,256))).astype('uint8')[::,::,::-1]
                style_save    =  (cv2.resize(style_256, (256,256))).astype('uint8')[::,::,::-1]
                bundle_result =  np.stack( (content_save, style_save, result), axis=1 )
                bundle_result = bundle_result.reshape((256, 256*3, 3))


                """ save the result """
                cv2.imwrite( os.path.join(save_dir_path, c_name+'.png'), content_save)
                cv2.imwrite( os.path.join(save_dir_path, s_name+'.png'), style_save)
                cv2.imwrite( os.path.join(save_dir_path, 'result.png'), result)
                cv2.imwrite( os.path.join(save_dir_path, c_name + '_' +  s_name + '_' + 'result_bundle.png'), bundle_result)

                # if args.denorm == True:
                #     warped_s2c_imgs= (np.clip( ( warped_s2c_imgs[0].clone().permute(1,2,0).detach().cpu().numpy()) *std + mean, 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
                # else:
                #     warped_s2c_imgs = (np.clip( ( warped_s2c_imgs[0].clone().permute(1,2,0).detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
                # resolution = ['256', '128', '64', '32', '16']
                # cv2.imwrite(  os.path.join(save_dir_path, 'warp_{0}.png'.format(resolution[2])) ,  warped_s2c_imgs ) # k=2 (64)

        print("time :", time.time() - start)
        print("root path: ", root_iter_path)

        del model 
        gc.collect()
