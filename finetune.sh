export CUDA_VISIBLE_DEVICES=7
export MODEL_PARENT_DIR=baichuan-inc
export MODEL_DIR=Baichuan2-7B-Chat
export MODEL_MAX_LENGTH=1024

export VERSION=v1genqkvo-v1gen-kb1
export LEARNING_RATE=2e-4
export NUM_TRAIN_EPOCHS=8
export GRADIENT_ACCUMULATION_STEPS=1
export WARMUP_RATIO=0.0
export DROPOUT=0.1
export LORA_R=32
export LORA_ALPHA=64
export SEED=42
export OUTPUT_DIR=Checkpoints/$MODEL_DIR/seed${SEED}/$VERSION

export MAX_NEW_TOKENS=32

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune.py \
    --model_name_or_path $MODEL_PARENT_DIR/$MODEL_DIR \
    --data_path data \
    --output_dir $OUTPUT_DIR \
    --model_max_length $MODEL_MAX_LENGTH \
    --version $VERSION \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy epoch \
    --eval_strategy epoch \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type constant \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --dropout $DROPOUT \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio $WARMUP_RATIO \
    --seed $SEED \
    --bf16 True \
    --max_new_tokens $MAX_NEW_TOKENS \
    --predict_with_generate True