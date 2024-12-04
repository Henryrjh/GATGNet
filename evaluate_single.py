
import argparse
import numpy as np
import torch, io
from TGRNet.TGRNet import create_TGRNet
import cv2
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def compute_rs_epe(flow_gt, flow_pr, valid_gt):
    # 将输入数据转换为 64 位浮点数进行计算
    flow_gt = flow_gt.astype(np.float64)
    flow_pr = flow_pr.astype(np.float64)
    valid_gt = valid_gt.astype(np.float64)

    error = np.sum(np.abs(flow_gt - flow_pr) * valid_gt)

    nums = np.sum(valid_gt)

    epe = error / nums

    return error, nums, epe

def compute_rs_d1(flow_gt, flow_pr, valid_gt, thresold = 3):

    # 将输入数据转换为 64 位浮点数进行计算
    flow_gt = flow_gt.astype(np.float64)
    flow_pr = flow_pr.astype(np.float64)
    valid_gt = valid_gt.astype(np.float64)

    err_map = np.abs(flow_gt - flow_pr) * valid_gt

    err_mask = err_map > thresold

    err_disps = np.sum(err_mask.astype('float64'))

    nums = np.sum(valid_gt)

    d1 = err_disps / nums

    return err_disps, nums, d1

def img_norm(img):
    mean = np.mean(img)
    std = np.std(img)
    new_img = (img - mean) / std
    return new_img

@torch.no_grad()
def validate_rs_single(model, left_path, right_path, save_path=None, mode='16bit', device='cuda', mixed_prec=False):
    """ Peform validation using the rs split """
    model.eval()
    model.to(device)
    # aug_params = {}
    left_img = cv2.imread(left_path, -1)
    right_img = cv2.imread(right_path, -1) 

    if mode == '16bit': ### WHU        
        assert (len(left_img.shape)==2), '16bit must be single channel!'
        left_img = np.tile(left_img[...,None], (1, 1, 3))
        right_img = np.tile(right_img[...,None], (1, 1, 3))
        left_img = img_norm(left_img)
        right_img = img_norm(right_img)
    elif mode == '8bit':    ### US3D
        assert (len(left_img.shape)==3) and (left_img.shape[2]==3), '8bit must be 3 channel!'
        left_img = left_img[..., :3]
        right_img = right_img[..., :3]
        left_img = left_img.astype(np.float32) / 255.
        right_img = right_img.astype(np.float32) / 255. 

    left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
    right_img = torch.from_numpy(right_img).permute(2, 0, 1).float()
    image1 = left_img[None].to(device)
    image2 = right_img[None].to(device)

    with autocast(enabled=mixed_prec):
        _, flow_pr = model(image1, image2, test_mode=True)
    flow_pr = flow_pr.float().cpu().squeeze(0)
    pre_disp = - np.array(flow_pr[0])

    if save_path is not None:
        cv2.imwrite(save_path, pre_disp)
        print(f'save predict disp: {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_path', default='KM_left_0.tiff')
    parser.add_argument('--right_path', default='KM_right_0.tiff')
    parser.add_argument('--save_path', default='KM_pred_0.tiff.tiff')
    parser.add_argument('--mode', default='16bit', choices=['8bit', '16bit'], help='The format of input data')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='cpu or cuda')

    args = parser.parse_args()

    model = create_TGRNet(args.mode)

    validate_rs_single(model, args.left_path, args.right_path, args.save_path, mode=args.mode, device=args.device)
