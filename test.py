import glob
import os, utils
from torch.utils.data import DataLoader
from models import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.MPLD import CONFIGS as CONFIGS_TM
import models.MPLD as MPLD
import SimpleITK as sitk

def main():
    test_dir = 'E:/XXX/testpklzuixin1'


    files = "E:/XXX/data/testniiBANBEN1"
    test_MR_file_lst = glob.glob(os.path.join(files, "*_MR.nii.gz"))
    test_CT_file_lst = glob.glob(os.path.join(files, "*_CT.nii.gz"))





    model_idx = -1
    weights = [1, 2]
    model_folder = 'MPLD_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder


    config = CONFIGS_TM['MPLD']
    model = MPLD.MPLD(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((32, 256, 384), 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    device = torch.device('cuda:0')
    atlas_pth = "E:/XXX/data/testniiBANBEN/img_1_CT.nii.gz"
    atlas = sitk.ReadImage(atlas_pth)
    DSC = []
    DSC_pre = []
    with torch.no_grad():
        # stdy_idx = 0
        for i in range(1, len(test_MR_file_lst)+1):
            # model.eval()
            # data = [t.cuda() for t in data]
            CT_file = test_CT_file_lst[i - 1]
            MR_file = test_MR_file_lst[i - 1]
            y = sitk.GetArrayFromImage(sitk.ReadImage(CT_file))[np.newaxis, np.newaxis, ...]
            # y = MinMax(y)
            y = torch.from_numpy(y).to(device).float()

            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(MR_file))[np.newaxis, np.newaxis, ...]
            print(input_moving.shape)

            # input_moving = MinMax(input_moving)
            x = torch.from_numpy(input_moving).to(device).float()

            moving_label = glob.glob(os.path.join(files, '*_MRlabel.nii.gz'))[i - 1]
            fixed_label = glob.glob(os.path.join(files, '*_CTlabel.nii.gz'))[i - 1]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label))[np.newaxis, np.newaxis, ...]
            input_label = MinMax(input_label)

            x_seg = torch.from_numpy(input_label).to(device).float()

            fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label))[np.newaxis, np.newaxis, ...]
            # print(fixed_label.shape)
            fixed_label = MinMax(fixed_label)
            y_seg = torch.from_numpy(fixed_label).to(device).float()

            # x = data1.to(device).float()
            # y = data2.to(device).float()
            # x_seg = data[2]
            # y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            grid_img = mk_grid_img(8, 1, config.img_size)
            def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
            # grid_fig = comput_fig(def_grid)

            dice_pre = compute_label_dice(x_seg.cpu().detach().numpy(), y_seg.cpu().detach().numpy())
            dice = compute_label_dice(y_seg.cpu().detach().numpy(), def_out.cpu().detach().numpy())
            print("dice_pre: ", dice_pre)
            print("dice: ", dice)
            DSC.append(dice)
            DSC_pre.append(dice_pre)
            if '6' in CT_file:
                save_image(def_grid, atlas, "61_grid.nii.gz")
                save_image(x_def, atlas, "61_warped.nii.gz")
                save_image(flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], atlas, "61_flow.nii.gz")
                save_image(def_out, atlas, "61_label.nii.gz")
            del x_def, flow, def_out

            # tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            # line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            # dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            y_def, flow = model(y_in)
            def_out = reg_model([y_seg.cuda().float(), flow.cuda()])

            dice_pre = compute_label_dice(x_seg.cpu().detach().numpy(), y_seg.cpu().detach().numpy())
            dice = compute_label_dice(y_seg.cpu().detach().numpy(), def_out.cpu().detach().numpy())
            print("dice_pre: ", dice_pre)
            print("dice: ", dice)
            DSC.append(dice)
            DSC_pre.append(dice_pre)
            if '8' in CT_file:
                save_image(y_def, atlas, "81_warpedMutualverse.nii.gz")
                save_image(flow, atlas, "81_flowMutualverse.nii.gz")
                save_image(def_out, atlas, "81_labelMutualverse.nii.gz")
            del y_def, flow, def_out


            # tar = x.detach().cpu().numpy()[0, 0, :, :, :]
            # jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            # line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            # line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            # out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            # print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # csv_writter(line, 'experiments/' + model_folder[:-1])
            # eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            # dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            # dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            # print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            # eval_dsc_def.update(dsc_trans.item(), x.size(0))
            # eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # stdy_idx += 1

        # print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def MinMax(image):
    image = image - np.mean(image)
    if np.max(image) - np.mean(image)!=0:
        image = (image - np.min(image)) / (np.max(image)-np.min(image))

    return image
def mk_grid_img(grid_step, line_thickness=1, grid_sz=(32, 256, 384)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img
def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1,2,3,4]
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
def save_image(img, ref_img, name):
    #img= img.numpy()
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())

    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join("E:/XXX/data/result", name))
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