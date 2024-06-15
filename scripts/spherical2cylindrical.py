import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool

dataset = '3D60'
root_path = '/path/to/3D60_Cassini'
save_path = '/path/to/3D60_Cy'
# this fov can get a size of 1024x512 for Deep360 and 512x256 for 3D60
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

def convert_equirectangular_to_cylindrical(equirectangular_image, vertical_fov=np.pi*120/180, scale=1, mode='nearest'):
    # Load equirectangular image
    equirectangular = Image.open(equirectangular_image)
    equirectangular = equirectangular.rotate(-90, expand=True)

    if scale == 1:
        width_equirectangular, height_equirectangular = equirectangular.size
    else:
        # resample
        width_equirectangular, height_equirectangular = equirectangular.size
        equirectangular = equirectangular.resize((width_equirectangular * scale, height_equirectangular * scale), Image.LANCZOS)
        width_equirectangular, height_equirectangular = equirectangular.size

    # Calculate horizontal FOV of equirectangular image
    horizontal_fov = 2 * np.pi

    # Calculate cylindrical width and height based on vertical FOV
    cylindrical_width = width_equirectangular
    cylindrical_height = 2 * int(height_equirectangular / np.pi * np.tan(vertical_fov / 2))

    # Create a blank cylindrical image
    cylindrical = Image.new("RGB", (cylindrical_width, cylindrical_height))

    for y_c in range(cylindrical_height):
        for x_c in range(cylindrical_width):
            # Calculate longitude and latitude
            theta = (x_c / cylindrical_width) * horizontal_fov - np.pi
            phi = np.arctan((y_c - cylindrical_height / 2) / (width_equirectangular / (2 * np.pi)))

            # Convert longitude and latitude to pixel coordinates in equirectangular image
            x_e = (theta + np.pi) / (2 * np.pi) * width_equirectangular
            y_e = (phi*2/np.pi +1) * height_equirectangular / 2

            if mode == 'nearest':
                y1 = int(np.round(y_e))
                x1 = int(np.round(x_e))
                # Get pixel value from equirectangular image
                pixel = equirectangular.getpixel((x1, y1))

            elif mode == 'bilinear':
                # bilinear interpolation
                # Get the four surrounding pixels in the original image
                x_original = (theta + np.pi) / (2 * np.pi) * width_equirectangular
                y_original = (phi * 2 / np.pi +1) * height_equirectangular / 2
                x_floor, y_floor = int(x_original), int(y_original)
                x_ceil, y_ceil = min(x_floor + 1, width_equirectangular - 1), min(y_floor + 1, height_equirectangular - 1)

                # Get the color values of the surrounding pixels
                tl = equirectangular.getpixel((x_floor, y_floor))
                tr = equirectangular.getpixel((x_ceil, y_floor))
                bl = equirectangular.getpixel((x_floor, y_ceil))
                br = equirectangular.getpixel((x_ceil, y_ceil))

                # Calculate the fractional parts for interpolation
                x_fraction = x_original - x_floor
                y_fraction = y_original - y_floor

                # Perform bilinear interpolation
                pixel = (
                    int((1 - x_fraction) * (1 - y_fraction) * tl[0] + x_fraction * (1 - y_fraction) * tr[0] +
                        (1 - x_fraction) * y_fraction * bl[0] + x_fraction * y_fraction * br[0]),
                    int((1 - x_fraction) * (1 - y_fraction) * tl[1] + x_fraction * (1 - y_fraction) * tr[1] +
                        (1 - x_fraction) * y_fraction * bl[1] + x_fraction * y_fraction * br[1]),
                    int((1 - x_fraction) * (1 - y_fraction) * tl[2] + x_fraction * (1 - y_fraction) * tr[2] +
                        (1 - x_fraction) * y_fraction * bl[2] + x_fraction * y_fraction * br[2])
                )

            # Set the pixel color in the resized image
            cylindrical.putpixel((x_c, y_c), pixel)
    
    if scale == 1:
        cylindrical = cylindrical.rotate(90, expand=True)
    else:
        cylindrical = cylindrical.resize((cylindrical_width // scale, cylindrical_height // scale), Image.NEAREST)
        cylindrical = cylindrical.rotate(90, expand=True)
    return cylindrical

def convert(item):
    left_image_path = item[0]
    right_image_path = item[1]

    cylindrical_left_image = convert_equirectangular_to_cylindrical(left_image_path, vertical_fov, scale=1, mode='bilinear')
    cylindrical_right_image = convert_equirectangular_to_cylindrical(right_image_path, vertical_fov, scale=1, mode='bilinear')

    left_save_path = left_image_path.replace(root_path, save_path)
    right_save_path = right_image_path.replace(root_path, save_path)

    left_save_folder, left_save_file = os.path.split(left_save_path)
    right_save_folder, right_save_file = os.path.split(right_save_path)

    os.makedirs(left_save_folder, exist_ok = True)
    os.makedirs(right_save_folder, exist_ok = True)

    cylindrical_left_image.save(left_save_path, "png")
    cylindrical_right_image.save(right_save_path, "png")

def main():
    if dataset == '3D60':
        list_val = list_3D60_disparity_all(root_path, '../dataloader/3d60_val.txt')
        list_train = list_3D60_disparity_all(root_path, '../dataloader/3d60_train.txt')
        list_test = list_3D60_disparity_all(root_path, '../dataloader/3d60_test.txt')
        list_all = list_val + list_train + list_test
    elif dataset == 'deep360':
        list_all = list_deep360_disparity_all(root_path, False)

    with Pool(processes=24) as p:
        max_ = len(list_all)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(convert, list_all):
                pbar.update()
        p.close()
        p.join()

if __name__ == "__main__":
    main()