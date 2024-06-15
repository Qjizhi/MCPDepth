python train_disparity.py \
                          --dataset Deep360 \
                          --dataset_root /path/to/Deep360_Cy/ \
                          --checkpoint_disp ./pretrained_model/pretrained_sceneflow_new.tar \
                          --loadSHGonly \
                          --parallel \
                          --max_disp 272 \
                          --save_checkpoint_path ./checkpoints/disp_deep360_cy \
                          --batch_size 1