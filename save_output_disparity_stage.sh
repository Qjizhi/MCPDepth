python save_output_disparity_stage.py \
    --projection cylindrical \
    --checkpoint_disp ./checkpoints_cy_circular_padding_dilation/disp/ModeDisparity/Deep360_Cy/ckpt_disp_ModeDisparity_Deep360_Cy_41.tar \
    --datapath /media/feng/2TB/dataset/Deep360_Cy_nearest_fov2 \
    --outpath ./outputs/Deep360PredDepthCylin/ \
    --max_disp 272 \
    --batch_size 1 \