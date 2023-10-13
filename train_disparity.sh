# python train_disparity.py \
#                           --dataset Deep360 \
#                           --dataset_root /media/feng/2TB/dataset/Deep360_Cy_nearest_fov2/ \
#                           --checkpoint_disp ./pretrained_model/pretrained_sceneflow_new.tar \
#                           --loadSHGonly \
#                           --parallel \
#                           --max_disp 272 \
#                           --save_checkpoint_path ./checkpoints_test/disp \
#                           --batch_size 1
python train_disparity.py \
                          --dataset 3D60 \
                          --dataset_root /media/feng/2TB/dataset/3D60_Cy/ \
                          --checkpoint_disp ./pretrained_model/pretrained_sceneflow_new.tar \
                          --loadSHGonly \
                          --parallel \
                          --max_disp 272 \
                          --save_checkpoint_path ./checkpoints_attention/disp \
                          --batch_size 1
