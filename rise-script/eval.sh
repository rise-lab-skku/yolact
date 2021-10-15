#! /bin/bash
common_args="--top_k=20 --score_threshold=0.50 --no_bar --fast_nms=True"
export CUDA_LAUNCH_BLOCKING=1

function my_eval {
    local weight_dir=$1
    local weight=$2
    local img_sz=$3
    local config_name=${weight_dir}_config
    local images_dir=data/coco/unloader/unloader_rgbd_20210930/color
    local depth_images_dir=data/coco/unloader/unloader_rgbd_20210930/depth

    # data/coco_ul_aug/test_resize/${img_sz}

    mkdir -p ./results/summary/${weight_dir}/image_result
    mkdir -p ./results/summary/${weight_dir}/prof
    echo "[${weight_dir}/${img_sz}] , img: ${img_sz} " >> ./results/summary/${weight_dir}/log.txt
    
    # uncomment this for qualitative results
    python3 ./eval.py --trained_model=./weights/${weight_dir}/${weight} --depth_images=${depth_images_dir} --images=${images_dir}:results/summary/${weight_dir}/image_result --config=${config_name} ${common_args}
    
    # uncomment this for mAP log
    python3 ./eval.py --trained_model=./weights/${weight_dir}/${weight} --config=${config_name} ${common_args} >>  ./results/summary/${weight_dir}/log.txt
    
    # uncomment this for benchmark log
    python3 ./eval.py --trained_model=./weights/${weight_dir}/${weight} --config=${config_name} ${common_args} --benchmark >>  ./results/summary/${weight_dir}/log.txt
    
    # uncomment this for profiler
    # python3 -m cProfile -o ./results/summary/${weight_dir}/prof/${img_sz}.prof ./eval.py --trained_model=./weights/${weight_dir}/${weight} --dataset=ul_aug_benchmark${img_sz} ${common_args} --config=${config_name} --benchmark
}

# my_eval max550_resnet101 yolact_base_178_10000.pth 550
# my_eval max550_resnet101 yolact_base_178_10000.pth 720
# my_eval max550_resnet101 yolact_base_178_10000.pth 1024
# my_eval max550_resnet101 yolact_base_178_10000.pth 2048

# my_eval max550 yolact_resnet50_422_190000.pth 550
# my_eval max550 yolact_resnet50_422_190000.pth 720
# my_eval max550 yolact_resnet50_422_190000.pth 1024
# my_eval max550 yolact_resnet50_422_190000.pth 2048

# my_eval max720 yolact_resnet50_155_70000.pth 550
# my_eval max720 yolact_resnet50_155_70000.pth 720
# my_eval max720 yolact_resnet50_155_70000.pth 1024
# my_eval max720 yolact_resnet50_155_70000.pth 2048

# my_eval max1024_resnet101 yolact_base_222_50000.pth 550
# my_eval max1024_resnet101 yolact_base_222_50000.pth 720
# my_eval max1024_resnet101 yolact_base_222_50000.pth 1024
# my_eval max1024_resnet101 yolact_base_222_50000.pth 2048

# my_eval max2048 yolact_resnet50_4878_200000.pth 550
# my_eval max2048 yolact_resnet50_4878_200000.pth 720
# my_eval max2048 yolact_resnet50_4878_200000.pth 1024
# my_eval max2048 yolact_resnet50_4878_200000.pth 2048

# my_eval yolact_resnet50_max1024 yolact_resnet50_max1024_config_892_100000.pth 2048
# my_eval yolact_resnet50_max1024 yolact_resnet50_max1024_config_892_100000.pth failed

# my_eval yolact_mobilenetv2_max1024 yolact_mobilenetv2_max1024_892_50000.pth 2048
# my_eval yolact_resnet50_max1024_depth_to_red yolact_resnet50_max1024_depth_to_red_3333_10000.pth 2048
# my_eval yolact_resnet50_max1024_bgrd16uc4 yolact_resnet50_max1024_bgrd16uc4_5038_15114_interrupt.pth 2048
# my_eval yolact_resnet50_wisdom yolact_resnet50_wisdom_1176_20000.pth 2048
my_eval yolact_resnet50_unloader_rgbd yolact_resnet50_unloader_rgbd_1964_110000.pth 2048

