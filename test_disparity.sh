CUDA_VISIBLE_DEVICES=0 python test_disparity.py \
                          --dataset Deep360 \
                          --dataset_root /media/sim/10TB/dataset/Deep360 \
                          --checkpoint_disp /home/sim/Documents/fq/MODE-2022/checkpoints/disp_sphe_regular/ckpt_disp_ModeDisparity_Deep360_55.tar \
                          --parallel \
                          --max_disp 272 \
                          --save_output_path eval/disp_sphe_regular