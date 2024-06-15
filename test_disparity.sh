CUDA_VISIBLE_DEVICES=0 python test_disparity.py \
                          --dataset Deep360 \
                          --dataset_root /path/to/Deep360_Cy/ \
                          --projection cylindrical \
                          --checkpoint_disp ./pretrained_model/MCPDepth_disparity_Deep360.tar \
                          --parallel \
                          --max_disp 272 \
                          --save_output_path eval/disp_deep360_cy
