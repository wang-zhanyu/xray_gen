cd /path/to/xray_gen
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

savepath="./save/v1"
llm_model="meta-llama/Llama-2-7b-chat-hf"
dataset='./data/mimic_cxr/annotation.json'
base_dir='./data/mimic_cxr'

python -u train.py \
    --dataset $dataset \
    --base_dir $base_dir \
    --llm_model $llm_model \
    --projection_dim 768 \
    --batch_size 16 \
    --val_batch_size 16 \
    --num_workers 8 \
    --learning_rate 0.0001 \
    --devices 1 \
    --strategy ddp \
    --max_epochs 30 \
    --accumulate_grad_batches 2 \
    --num_sanity_val_steps 2 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --savedmodel_path ${savepath} \
    2>&1 |tee -a ${savepath}/log.txt
