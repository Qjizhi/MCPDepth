import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool

dataset = '3D60'
root_path = '/media/feng/2TB/dataset/3D60_Cassini'
save_path = '/media/feng/2TB/dataset/3D60_Cy'
# this fov can get a size of 1024x512 
vertical_fov = 2 * np.arctan(np.pi/2)  # np.pi * 120/ 180

def list_deep360_disparity_all(filepath, soiled):
    
    list_all = []

    ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
    ep_list.sort()

    rgb_dir = "rgb_soiled" if soiled else "rgb"

    for ep in ep_list:
        for subset in ['training', 'validation', 'testing']:
            rgb_path = os.path.join(filepath, ep, subset, rgb_dir)
            disp_path = os.path.join(filepath, ep, subset, "disp")

            rgb_name_list = os.listdir(rgb_path)
            rgb_name_list.sort()
            disp_name_list = os.listdir(disp_path)
            disp_name_list.sort()


            for i in range(len(disp_name_list)):
                list_all.append([os.path.join(rgb_path, rgb_name_list[i * 2]), \
                                 os.path.join(rgb_path, rgb_name_list[i * 2 + 1]), \
                                 os.path.join(disp_path, disp_name_list[i])])

    return list_all

def getFileList(filenamespath):
    fileNameList = []
    with open(filenamespath) as f:
        lines = f.readlines()
        for line in lines:
            fileNameList.append(line.strip().split(" "))  # split by space
    return fileNameList

def list_3D60_disparity_all(filepath, filenamespath):
    list_all = []

    file_list = getFileList(filenamespath)
    for file in file_list:
        if file[0].split('/')[1] == 'Matterport3D':
            scene = file[0].split('/')[1]
            id = file[0].split('/')[-1].split('_')[0] + '_' + file[0].split('/')[-1].split('_')[1]
        elif file[0].split('/')[1] == 'Stanford2D3D':
            scene = file[0].split('/')[1] + '/' + file[0].split('/')[2]
            id = file[0].split('/')[-1].split('_')[0] + '_' + file[0].split('/')[-1].split('_')[1] + '_' + file[0].split('/')[-1].split('_')[2]
        elif file[0].split('/')[1] == 'SunCG':
            scene = file[0].split('/')[1]
            id = file[0].split('/')[-1].split('_')[0]
        for pair in ['lr', 'ud', 'ur']:
            left = id + '_color_0_' + pair + '_' + pair[0] + '_0.0.png'
            right = id + '_color_0_' + pair + '_' + pair[1] + '_0.0.png'
            disp = id + '_disp_0_' + pair + '_' + pair[0] + '_0.0.npz'
            left_flip = id + '_color_0_' + pair + '_' + pair[1] + '_0.0.png'
            right_flip = id + '_color_0_' + pair + '_' + pair[0] + '_0.0.png'
            disp_flip = id + '_disp_0_' + pair + '_' + pair[1] + '_0.0.npz'
            list_all.append([os.path.join(filepath, scene, left), \
                             os.path.join(filepath, scene, right), \
                             os.path.join(filepath, scene, disp)])
            list_all.append([os.path.join(filepath, scene, left_flip), \
                             os.path.join(filepath, scene, right_flip), \
                             os.path.join(filepath, scene, disp_flip)])
    return list_all


def convert_equirectangular_to_cylindrical_disp(equirectangular_image, vertical_fov, scale=1, mode='nearest'):
    # Load equirectangular image
    key = np.load(equirectangular_image).files[0]
    equirectangular = np.load(equirectangular_image)[key].astype(np.float32)
    equirectangular = np.rot90(equirectangular, -1)
    
    if scale == 1:
        height_equirectangular, width_equirectangular = equirectangular.shape

    else:
        # resample
        equirectangular = Image.fromarray(equirectangular)
        height_equirectangular, width_equirectangular = equirectangular.shape
        equirectangular = equirectangular.resize((width_equirectangular * scale, height_equirectangular * scale), Image.LANCZOS)
        # equirectangular = equirectangular.resize((width_equirectangular * scale, height_equirectangular * scale), Image.NEAREST)
        equirectangular = np.array(equirectangular) * scale
        height_equirectangular, width_equirectangular = equirectangular.shape

    # Calculate horizontal FOV of equirectangular image
    horizontal_fov = 2 * np.pi

    # Calculate cylindrical width and height based on vertical FOV
    cylindrical_width = width_equirectangular
    cylindrical_height = 2 * int(height_equirectangular / np.pi * np.tan(vertical_fov / 2))

    # Create a blank cylindrical image
    cylindrical = np.zeros((cylindrical_height, cylindrical_width))

    for y_c in range(cylindrical_height):
        for x_c in range(cylindrical_width):
            # Calculate longitude and latitude
            theta = (x_c / cylindrical_width) * horizontal_fov - np.pi
            phi = np.arctan((y_c - cylindrical_height / 2) / (width_equirectangular / (2 * np.pi)))

            # Convert longitude and latitude to pixel coordinates in equirectangular image
            x_e = (theta + np.pi) / (2 * np.pi) * width_equirectangular
            y_e = (phi * 2 / np.pi +1) * height_equirectangular / 2
            # nearest
            if mode == 'nearest':
                y1 = int(np.round(y_e))# if (int(np.round(y_e)) < height_equirectangular) else (height_equirectangular -1)
                x1 = int(np.round(x_e))# if (int(np.round(x_e)) < width_equirectangular) else (width_equirectangular - 1)
                equirectangular_disp = equirectangular[y1, x1]

            # bilinear interpolation to get equirectangular_disp
            elif mode == 'bilinear':
                x1 = int(x_e)
                y1 = int(y_e)
                if x1 >= cylindrical_width - 1:
                    y2 = y1 + 1
                    fy = y_e - y1
                    q11 = equirectangular[y1, x1]
                    q12 = equirectangular[y2, x1]
                    equirectangular_disp = (1 - fy) * q11 + fy * q12
                elif y1 >= cylindrical_height - 1:
                    x2 = x1 + 1
                    fx = x_e - x1
                    q11 = equirectangular[y1, x1]
                    q21 = equirectangular[y1, x2]
                    equirectangular_disp = (1 - fx) * q11 + fx * q21
                else:
                    x2 = x1 + 1
                    y2 = y1 + 1
                    fx = x_e - x1
                    fy = y_e - y1
                    q11 = equirectangular[y1, x1]
                    q21 = equirectangular[y1, x2]
                    q12 = equirectangular[y2, x1]
                    q22 = equirectangular[y2, x2]
                    equirectangular_disp = (1 - fx) * (1 - fy) * q11 + fx * (1 - fy) * q21 + (1 - fx) * fy * q12 + fx * fy * q22
            else:
                print("Wrong mode!")

            focal_length = width_equirectangular / (2 * np.pi)
            phi_right = phi - equirectangular_disp * np.pi / height_equirectangular + np.pi / 2
            y_c_right = -(focal_length / np.tan(phi_right)) + cylindrical_height / 2

            cylindrical[y_c, x_c] = y_c - y_c_right

    if scale == 1:
        cylindrical = np.rot90(cylindrical, 1)
    else:
        cylindrical = Image.fromarray(cylindrical)
        cylindrical = cylindrical.resize((cylindrical_width // scale, cylindrical_height // scale), Image.NEAREST)
        cylindrical = np.array(cylindrical)
        cylindrical = np.rot90(cylindrical, 1) / scale
    return cylindrical

def convert(item):
    cylindrical_left_disp = convert_equirectangular_to_cylindrical_disp(item[2], vertical_fov, mode='nearest')

    left_save_path = item[2].replace(root_path, save_path)
    left_save_folder, left_save_file = os.path.split(left_save_path)
    os.makedirs(left_save_folder, exist_ok = True)

    np.savez_compressed(left_save_path, data=cylindrical_left_disp)

def main():
    if dataset == '3D60':
        list_val = list_3D60_disparity_all(root_path, '../dataloader/3d60_val.txt')
        list_train = list_3D60_disparity_all(root_path, '../dataloader/3d60_train.txt')
        list_test = list_3D60_disparity_all(root_path, '../dataloader/3d60_test.txt')
        list_all = list_val + list_train + list_test
    elif dataset == 'deep360':
        list_all = list_deep360_disparity_all(root_path, False)

    with Pool(processes=48) as p:
        max_ = len(list_all)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(convert, list_all):
                pbar.update()
        p.close()
        p.join()

if __name__ == "__main__":
    main()