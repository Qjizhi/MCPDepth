CUDA_VISIBLE_DEVICES=0 python test_disparity.py \
                          --dataset Deep360 \
                          --dataset_root /media/feng/2TB/dataset/Deep360_Cy \
                          --projection cylindrical \
                          --checkpoint_disp /home/feng/Documents/autel_fq/MODE-2022/checkpoints_cy_circular_padding_dilation/disp/ModeDisparity/Deep360_Cy/ckpt_disp_ModeDisparity_Deep360_Cy_41.tar \
                          --parallel \
                          --max_disp 272 \
                          --save_output_path eval/disp_cy
# CUDA_VISIBLE_DEVICES=0 python test_disparity.py \
#                           --dataset Deep360 \
#                           --dataset_root /media/feng/2TB/dataset/Deep360 \
#                           --projection cassini \
#                           --checkpoint_disp ./pretrained_model/ckpt_disp_ModeDisparity_Deep360.tar \
#                           --parallel \
#                           --max_disp 192 \
#                           --save_output_path eval/disp