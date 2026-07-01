tport=52009
#ngpu=4
ngpu=2
ROOT=/home/chicken/Code/AugSeg-main2

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=${ngpu} \
    --node_rank=0 \
    --master_port=${tport} \
    $ROOT/train_semi.py \
    --config=$ROOT/exps/zrun_citys/citys_semi744/config_semi.yaml --seed 2 --port ${tport}

# --- --- --- 
    # --config=$ROOT/exps/zrun_citys/citys_semi186/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi372/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi744/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/citys_semi1488/config_semi.yaml --seed 2 --port ${tport}

    # --config=$ROOT/exps/zrun_citys/r50_citys_semi186/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi372/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi744/config_semi.yaml --seed 2 --port ${tport}
    # --config=$ROOT/exps/zrun_citys/r50_citys_semi1488/config_semi.yaml --seed 2 --port ${tport}
