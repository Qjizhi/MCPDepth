python train_disparity.py \
                          --dataset Deep360 \
                          --dataset_root /media/feng/2TB/dataset/Deep360_Cy_nearest_fov2/ \
                          --checkpoint_disp ./pretrained_model/pretrained_sceneflow_new.tar \
                          --loadSHGonly \
                          --parallel \
                          --max_disp 272 \
                          --save_checkpoint_path ./checkpoints_attention/disp \
                          --batch_size 1