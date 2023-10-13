from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataloader import list_deep360_disparity_train, list_deep360_disparity_test
from dataloader import Deep360DatasetDisparity
from models import ModeDisparity
from utils.geometry import rotateCassini, depthViewTransWithConf
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description='MODE - save disparity and confidence outputs')
parser.add_argument('--projection', choices=['cylindrical', 'cassini'], help='model trained in which projection.')
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--dbname', default='Deep360', help='dataset name')
parser.add_argument('--datapath', default='../../datasets/Deep360/', help='datapath')
parser.add_argument('--soiled', action='store_true', default=False, help='output the intermediate results of soiled dataset')
parser.add_argument('--outpath', default='./outputs/Deep360PredDepth/', help='output path')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--checkpoint_disp', default=None, help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
  train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp = list_deep360_disparity_train(args.datapath, args.soiled)
  test_left_img, test_right_img, test_left_disp = list_deep360_disparity_test(args.datapath, args.soiled)
  total_num = (len(train_left_img) + len(val_left_img) + len(test_left_img))
  # list all files---------------------------------
  # left
  all_left = train_left_img
  all_left.extend(val_left_img)
  all_left.extend(test_left_img)
  # right
  all_right = train_right_img
  all_right.extend(val_right_img)
  all_right.extend(test_right_img)
  # disp
  all_disp = train_left_disp
  all_disp.extend(val_left_disp)
  all_disp.extend(test_left_disp)
  assert (len(all_left) == len(all_right) == len(all_disp) == total_num)
  #------------------------------------------------

AllImgLoader = torch.utils.data.DataLoader(Deep360DatasetDisparity(all_left, all_right, all_disp), batch_size=args.batch_size, shuffle=False, num_workers=args.batch_size, drop_last=False)

# Note
# in_height,in_width: shape of input image. using (1024,512) for deep360
if args.dbname == 'Deep360':
  model = ModeDisparity(args.max_disp, conv='Sphere', in_height=1024, in_width=512, out_conf=True)

if args.cuda:
  model = nn.DataParallel(model)
  model.cuda()

if args.checkpoint_disp is not None:
  print('Load pretrained model')
  pretrain_dict = torch.load(args.checkpoint_disp)
  model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def cylindrical_to_spherical_disp(cylindrical_img, vertical_fov=np.pi*120/180, scale=1, mode='nearest'):
    # Read and rotate
    # key = np.load(cylindrical_img).files[0]
    # cylindrical_img = np.load(cylindrical_img)[key].astype(np.float32)
    cylindrical_img = np.rot90(cylindrical_img, -1)

    if scale == 1:
        # Get the input cylindrical image dimensions
        input_height, input_width = cylindrical_img.shape
        output_height = 2 * int(vertical_fov / 2 * input_width / (2*np.pi))
    else:
        # resample
        cylindrical_img = Image.fromarray(cylindrical_img)
        cylindrical_img = cylindrical_img.resize((input_width * scale, input_height * scale), Image.NEAREST)
        cylindrical_img = np.array(cylindrical_img) * scale
        input_height, input_width = cylindrical_img.shape
        output_height = scale * 2 * int(vertical_fov / 2 * input_width / (2*np.pi) / scale)

    output_width = input_width
    spherical_img = np.zeros((output_height, output_width))

    # Calculate the vertical and horizontal field of view in radians
    fov_vertical = vertical_fov
    fov_horizontal = 2 * np.pi

    # Loop over each pixel in the output spherical image
    for y_out in range(output_height):
        for x_out in range(output_width):
            # Calculate longitude and latitude
            theta = (x_out / output_width) * fov_horizontal - np.pi
            phi =  (y_out - output_height /2) / (input_width / (2*np.pi))

            # bilinear interpolation
            # Get the four surrounding pixels in the original image
            x_e = (theta + np.pi) / (2 * np.pi) * output_width
            y_e = (input_width / (2*np.pi)) * np.tan(phi) + (input_height / 2)

            # nearest
            if mode == 'nearest':
                y1 = int(np.round(y_e)) if (int(np.round(y_e)) < input_height) else (input_height -1)
                x1 = int(np.round(x_e)) if (int(np.round(x_e)) < input_width) else (input_width -1)
                cylindrical_disp = cylindrical_img[y1, x1]

            # bilinear interpolation to get cylindrical disparity
            elif mode == 'bilinear':
                x1 = int(x_e)
                y1 = int(y_e)

                if x1 >= input_width - 1:
                    y2 = y1 + 1
                    fy = y_e - y1
                    q11 = cylindrical_img[y1, x1]
                    q12 = cylindrical_img[y2, x1]
                    cylindrical_disp = (1 - fy) * q11 + fy * q12
                elif y1 >= input_height - 1:
                    x2 = x1 + 1
                    fx = x_e - x1
                    q11 = cylindrical_img[y1, x1]
                    q21 = cylindrical_img[y1, x2]
                    cylindrical_disp = (1 - fx) * q11 + fx * q21
                else:
                    x2 = x1 + 1
                    y2 = y1 + 1
                    fx = x_e - x1
                    fy = y_e - y1
                    q11 = cylindrical_img[y1, x1]
                    q21 = cylindrical_img[y1, x2]
                    q12 = cylindrical_img[y2, x1]
                    q22 = cylindrical_img[y2, x2]
                    cylindrical_disp = (1 - fx) * (1 - fy) * q11 + fx * (1 - fy) * q21 + (1 - fx) * fy * q12 + fx * fy * q22
            else:
                print("Wrong mode!")

            focal_length = input_width / (2*np.pi)
            y_c_right = y_e - cylindrical_disp
            phi_right = np.arctan((y_c_right - input_height / 2) / focal_length)
            spherical_disp = (phi - phi_right) * focal_length
            spherical_img[y_out, x_out] = spherical_disp

    if scale == 1:
        spherical_img = np.rot90(spherical_img, 1)
    else:
        spherical_img = Image.fromarray(spherical_img)
        spherical_img = spherical_img.resize((output_width // scale, output_height // scale), Image.NEAREST)
        spherical_img = np.array(spherical_img)
        spherical_img = np.rot90(spherical_img, 1) / scale
    spherical_img = np.ascontiguousarray(spherical_img)
    return spherical_img

def cylindrical_to_spherical(cylindrical_img, vertical_fov=np.pi*120/180, scale=1, mode='nearest'):
    # Read and rotate
    cylindrical_img = np.rot90(cylindrical_img, -1)

    # Get the input cylindrical image dimensions
    input_height, input_width = cylindrical_img.shape
    output_height = 2 * int(vertical_fov / 2 * input_width / (2*np.pi))


    output_width = input_width
    spherical_img = np.zeros((output_height, output_width))

    # Calculate the vertical and horizontal field of view in radians
    fov_vertical = vertical_fov
    fov_horizontal = 2 * np.pi

    # Loop over each pixel in the output spherical image
    for y_out in range(output_height):
        for x_out in range(output_width):
            # Calculate longitude and latitude
            theta = (x_out / output_width) * fov_horizontal - np.pi
            phi =  (y_out - output_height /2) / (input_width / (2*np.pi))

            # bilinear interpolation
            # Get the four surrounding pixels in the original image
            x_e = (theta + np.pi) / (2 * np.pi) * output_width
            y_e = (input_width / (2*np.pi)) * np.tan(phi) + (input_height / 2)

            # nearest
            if mode == 'nearest':
                y1 = int(np.round(y_e)) if (int(np.round(y_e)) < input_height) else (input_height -1)
                x1 = int(np.round(x_e)) if (int(np.round(x_e)) < input_width) else (input_width -1)
                cylindrical_disp = cylindrical_img[y1, x1]

            # bilinear interpolation to get cylindrical disparity
            elif mode == 'bilinear':
                x1 = int(x_e)
                y1 = int(y_e)

                if x1 >= input_width - 1:
                    y2 = y1 + 1
                    fy = y_e - y1
                    q11 = cylindrical_img[y1, x1]
                    q12 = cylindrical_img[y2, x1]
                    cylindrical_disp = (1 - fy) * q11 + fy * q12
                elif y1 >= input_height - 1:
                    x2 = x1 + 1
                    fx = x_e - x1
                    q11 = cylindrical_img[y1, x1]
                    q21 = cylindrical_img[y1, x2]
                    cylindrical_disp = (1 - fx) * q11 + fx * q21
                else:
                    x2 = x1 + 1
                    y2 = y1 + 1
                    fx = x_e - x1
                    fy = y_e - y1
                    q11 = cylindrical_img[y1, x1]
                    q21 = cylindrical_img[y1, x2]
                    q12 = cylindrical_img[y2, x1]
                    q22 = cylindrical_img[y2, x2]
                    cylindrical_disp = (1 - fx) * (1 - fy) * q11 + fx * (1 - fy) * q21 + (1 - fx) * fy * q12 + fx * fy * q22
            else:
                print("Wrong mode!")

            spherical_img[y_out, x_out] = cylindrical_disp

    spherical_img = np.rot90(spherical_img, 1)
    spherical_img = np.ascontiguousarray(spherical_img)
    return spherical_img

def output_disp_and_conf(imgL, imgR):
  model.eval()

  if args.cuda:
    imgL, imgR = imgL.cuda(), imgR.cuda()

  if imgL.shape[2] % 16 != 0:
    times = imgL.shape[2] // 16
    top_pad = (times + 1) * 16 - imgL.shape[2]
  else:
    top_pad = 0

  if imgL.shape[3] % 16 != 0:
    times = imgL.shape[3] // 16
    right_pad = (times + 1) * 16 - imgL.shape[3]
  else:
    right_pad = 0

  imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
  imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

  with torch.no_grad():
    output3, conf_map = model(imgL, imgR)
    output3 = torch.squeeze(output3, 1)
    conf_map = torch.squeeze(conf_map, 1)

  if top_pad != 0:
    output3 = output3[:, top_pad:, :]
  if right_pad != 0:
    output3 = output3[:, :, :-right_pad]

  return output3.data.cpu().numpy(), conf_map.data.cpu().numpy()


def disp2depth(disp, conf_map, cam_pair):
  cam_pair_dict = {'12': 0, '13': 1, '14': 2, '23': 3, '24': 4, '34': 5}

  if args.dbname == 'Deep360':
    baseline = np.array([1, 1, math.sqrt(2), math.sqrt(2), 1, 1]).astype(np.float32)
  elif args.dbname == '3D60':
    pass
  else:
    baseline = np.array([0.6 * math.sqrt(2), 0.6 * math.sqrt(2), 1.2, 1.2, 0.6 * math.sqrt(2), 0.6 * math.sqrt(2)]).astype(np.float32)

  output_h = disp.shape[0]
  output_w = disp.shape[1]

  phi_l_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_l_end = -0.5 * math.pi
  phi_l_step = math.pi / output_w
  phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
  phi_l_map = np.array([phi_l_range for j in range(output_h)]).astype(np.float32)

  mask_disp_is_0 = disp == 0
  disp_not_0 = np.ma.array(disp, mask=mask_disp_is_0)

  phi_r_map = disp_not_0 * math.pi / output_w + phi_l_map

  # sin theory
  depth_l = baseline[cam_pair_dict[cam_pair]] * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
  depth_l = depth_l.filled(1000)
  depth_l[depth_l > 1000] = 1000
  depth_l[depth_l < 0] = 0

  if cam_pair == '12':
    return depth_l, conf_map
  elif cam_pair == '13':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = rotateCassini(depth_1, 0.5 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = rotateCassini(conf_1, 0.5 * math.pi, 0, 0)
    return depth_2[:, :, 0], conf_2[:, :, 0]
  elif cam_pair == '14':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = rotateCassini(depth_1, 0.25 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = rotateCassini(conf_1, 0.25 * math.pi, 0, 0)
    return depth_2[:, :, 0], conf_2[:, :, 0]
  elif cam_pair == '23':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, -math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.75 * math.pi, 0, 0)
    return depth_2, conf_2
  elif cam_pair == '24':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, -1, 0, 0.5 * math.pi, 0, 0)
    return depth_2, conf_2
  elif cam_pair == '34':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, 1, 0, 0, 0, 0)
    return depth_2, conf_2
  else:
    print("Error! Wrong Cam_pair!")
    return None


def main():
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")
  if args.dbname == 'Deep360':
    #------------- Check the output directories -------
    ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
    ep_list.sort()
    outdir_name = "disp_pred2depth" if not args.soiled else "disp_pred2depth_soiled"
    outdir_conf_name = "conf_map" if not args.soiled else "conf_map_soiled"
    for ep in ep_list:
      for subset in ['training', 'validation', 'testing']:
        disp_pred2depth_path = os.path.join(args.outpath, ep, subset, outdir_name)
        os.makedirs(disp_pred2depth_path, exist_ok=True)
        conf_map_path = os.path.join(args.outpath, ep, subset, outdir_conf_name)
        os.makedirs(conf_map_path, exist_ok=True)
    #----------------------------------------------------------
  for batchIdx, batchData in enumerate(AllImgLoader):
    print("\rDisparity output progress: {:.2f}%".format(100 * (batchIdx + 1) / len(AllImgLoader)), end='')
    leftImg = batchData['leftImg'].to(device)
    rightImg = batchData['rightImg'].to(device)
    dispName = batchData['dispNames']

    pred_disp_batch, conf_map_batch = output_disp_and_conf(leftImg, rightImg)

    for i in range(pred_disp_batch.shape[0]):
      outpath = dispName[i].replace(args.datapath, args.outpath)
      outpath = outpath[:-8]
      #------------- save disp_pred ------------------
      # outpath_disp = outpath.replace("rgb","disp_pred")
      # np.save(outpath_disp + "disp_pred.npy", pred_disp_batch[i])
      #------------- do disp2depth ------------------
      # spherical
      if args.projection == 'cassini':
        depth_at_1, conf_at_1 = disp2depth(pred_disp_batch[i], conf_map_batch[i], dispName[i][-11:-9])
        outpath_depth = outpath.replace("disp", outdir_name)
        np.savez(outpath_depth + "disp_pred2depth.npz", depth_at_1)  #save npz files
        #------------- save conf_map ------------------
        outpath_conf = outpath.replace("disp", outdir_conf_name)
        cv2.imwrite(outpath_conf + "conf_map.png", conf_at_1 * 255)
      # cylindrical
      else:
        disp_cy, conf_cy = pred_disp_batch[i], conf_map_batch[i]
        disp_sph = cylindrical_to_spherical_disp(disp_cy, vertical_fov=2 * np.arctan(np.pi/2), mode='nearest')
        # TODO: support 3D60 size
        padding_size = (512 - disp_sph.shape[1]) // 2
        disp_sph = np.pad(disp_sph, ((0, 0), (padding_size, padding_size)), mode='constant', constant_values=0.0)
        conf_sph = cylindrical_to_spherical(conf_cy, vertical_fov=2 * np.arctan(np.pi/2), mode='bilinear')
        conf_sph = np.pad(conf_sph, ((0, 0), (padding_size, padding_size)), mode='constant', constant_values=0.0)
        depth_at_1, conf_at_1 = disp2depth(disp_sph, conf_sph, dispName[i][-11:-9])
        outpath_depth = outpath.replace("disp", outdir_name)
        np.savez(outpath_depth + "disp_pred2depth.npz", depth_at_1)  #save npz files
        #------------- save conf_map ------------------
        outpath_conf = outpath.replace("disp", outdir_conf_name)
        cv2.imwrite(outpath_conf + "conf_map.png", conf_at_1 * 255)
if __name__ == '__main__':
  main()
