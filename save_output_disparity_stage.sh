python save_output_disparity_stage.py \
    --projection cylindrical \
    --checkpoint_disp ./pretrained_model/MCPDepth_disparity_Deep360.tar \
    --datapath /path/to/Deep360_Cy/ \
    --outpath ./outputs/Deep360PredDepthCylin/ \
    --max_disp 272 \
    --batch_size 1 \