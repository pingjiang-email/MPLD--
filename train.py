import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os,  glob
import sys
import SimpleITK as sitk
from torch.utils.data import DataLoader
from models import datasets, trans,similarity_Evaluator
import numpy as np
import torch.utils.data as Data
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import models.utils as utils
from natsort import natsorted
from models.MPLD import CONFIGS as CONFIGS_TM
import models.MPLD as MPLD
from models.datagenerators import Dataset
# from models.MinMax import MinMax

# pythonCopy codedef_out  = None
# 使用str变量
# ...
def MinMax(image):
    image = image - np.mean(image)
    if np.max(image) - np.mean(image)!=0:
        image = (image - np.min(image)) / (np.max(image)-np.min(image))

    return image

# def_out = []

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    # train_dir = 'E:/GONGZUO/TUIXIANGPEIZH/PING/CHENGXU/IMSE-main/IMSE-3D-Pure/data/normimagesTS3/'
    moving_files = "E:/XXX/data/trainmoving1"
    fixed_files = "E:/XXX/data/trainfixed1"
    moving_file_lst = glob.glob(os.path.join(moving_files, "*.nii.gz"))
    fixed_file_lst = glob.glob(os.path.join(fixed_files,"*.nii.gz"))

    val_dir = 'E:/XXX/data/testpklzuixin1'
    # test_MR_file_lst = glob.glob(os.path.join( val_dir, '*_MR.nii.gz'))
    # test_CT_file_lst = glob.glob(os.path.join( val_dir, '*_CT.nii.gz'))

    save_dir = 'E:/XXX/'

    if not os.path.exists(save_dir+'experiments'):
        os.makedirs(save_dir+'experiments')
    if not os.path.exists(save_dir+'logs'  ):
        os.makedirs( save_dir + 'logs' )
    sys.stdout = Logger( save_dir + 'logs/' )

    # self.MR_files_root = glob.glob(os.path.join(root, '*_MR.nii.gz'))
    # self.CT_files_root = glob.glob(os.path.join(root, '*_CT.nii.gz'))
    # DS = Dataset(CT_images=train_CT_files, MR_images=train_MR_files)
    # print("Number of training images: ", len(DS))
    # DL = Data.DataLoader(DS, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    # device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    atlas_pth = "E:/XXX/data/trainmoving/img_0_MR.nii.gz"
    atlas = sitk.ReadImage(atlas_pth)

    weights = [1, 2] # loss weights
    # save_dir = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    # if not os.path.exists('experiments/'+save_dir):
    #     os.makedirs('experiments/'+save_dir)
    # if not os.path.exists('logs/'+save_dir):
    #     os.makedirs('logs/'+save_dir)
    # sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 1000 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['MPLD']
    model = MPLD.MPLD(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 1500
        model_dir = save_dir+'experiments/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        model_dict = "E:/XXX/LocalMI1.tar"
        # best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        best_model = torch.load(model_dict)
        # print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        # model.load_state_dict(best_model)
        print('Model loaded' + model_dict)
        model.load_state_dict(best_model['state_dict'])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip(0),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(), #rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16)),
                                        ])


    # val_set = datasets.JHUBrainDataset(glob.glob(val_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets.PairedImageDataset(val_dir)
    DS = datasets.Dataset2(file1=moving_file_lst, file2=fixed_file_lst)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    DL = DataLoader(DS, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    # print(len(DS))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)



    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
  
    criterion = similarity_Evaluator.OUR()

    criterions = [criterion]
    criterions += [similarity_Evaluator.Grad3d(penalty='l1')]
    best_dsc = 0
    writer = SummaryWriter(log_dir=save_dir+'logs1')

    device = torch.device('cuda:0')
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        # idx = 0
        for idx in range(1, len(DS)+ 1):
            input_moving, input_fixed, index1, index2 = next(iter(DL))
            # idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # input_moving=input_moving.cuda()
            # input_fixed=input_fixed.cuda()

            np_moving = input_moving.cpu().numpy()
            np_fixed = input_fixed.cpu().numpy()

            input_fixed = MinMax(np_fixed)
            input_moving = MinMax(np_moving)

            input_fixed = torch.from_numpy(input_fixed)
            input_moving = torch.from_numpy(input_moving)
            # print(input_fixed.shape)
            # print(input_moving.shape)

            x = input_fixed.to(device).float()
            y = input_moving.to(device).float()
            # data = [t.cuda() for t in data]
            # x = data[0]
            # y = data[1]
            x_in = torch.cat((x,y), dim=1)
            output = model(x_in)
            loss = 0
            loss_vals = []



            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
                # train_loss = []
                # train_loss.append(loss.item())
                # with open("./train_loss.txt", 'w') as train_los:
                #     train_los.write(str(train_loss))
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            # print(x_in.shape, x_in.dtype)
            output = model(x_in)


            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(DL), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data1, data2, x_seg, y_seg in val_loader:
                model.eval()
                # data = [t.cuda() for t in data]
                # print(data.shape)
                x = data1.to(device).float()
                y = data2.to(device).float()
                # x_seg = data[2]
                # y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                # print(x_in.shape, x_in.dtype)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                # 浮动场
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])

                # X的label和浮动场进行处理
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                pred_img = reg_model([y, output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.cuda().long(), 46)
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir+'experiments1/', filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))

        # validation
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        inputfixed = comput_fig(x)
        inputmoving = comput_fig(y)
        predimg = comput_fig(pred_img)
        # pred_fig = np.float32(torch.squeeze(def_out).cpu())
        # grid_fig = np.float32(torch.squeeze(def_grid).cpu())
        # x_fig = np.float32(torch.squeeze(x_seg).cpu())
        # tar_fig = np.float32(torch.squeeze(y_seg).cpu())
        writer.add_figure('inputfixed', inputfixed, epoch)
        plt.close(inputfixed)
        writer.add_figure('inputmoving', inputmoving, epoch)
        plt.close(inputmoving)
        writer.add_figure('predimg', predimg, epoch)
        plt.close(predimg)
        writer.add_figure('Grid', grid_fig, epoch)
        # plt.imshow()
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)


        loss_all.reset()
    writer.close()


def comput_fig(img):
    # global def_out
    # global pred_fig
    # global img
    # img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    img = img.detach().cpu().numpy()[0, 0, 8:24, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def save_image(img, ref_img, name):
    #img= img.numpy()
    # img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())

    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join("E:/XXX/data/result", name))



def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(32, 256, 384)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()