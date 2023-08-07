#!/bin/bash
save_precision="fp16" # 保存精度，默认为float
precision="float" # 合并时计算精度，推荐使用float
new_rank=4 # 输出LoRA模型的维度等级，默认为4
models="./output/modelA.safetensors ./output/modelB.safetensors" # 需要调整大小并合并的原始LoRA模型路径，保存为cpkt或safetensors格式，多个用空格隔开
ratios="1.0 -1.0" # 每个模型的合并比例，LoRA模型数量和比例数量一一对应，多个用空格隔开
save_to="./output/lora_name_new.safetensors" # 输出LoRA模型的路径，保存为ckpt或safetensors格式
device="cuda" # 使用的设备，cuda表示使用GPU，默认为CPU
new_conv_rank=0 # 指定Conv2d 3x3的输出等级，如果为0，则默认与new_rank相同
export HF_HOME="huggingface"
export XFORMERS_FORCE_DISABLE_TRITON="1"
# 创建一个数组用于存储附加的命令行参数
ext_args=()
ext_args+=("--models")
for model in $models; do
    ext_args+=("$model")
done
ext_args+=("--ratios")
for ratio in $ratios; do
    ext_args+=("$ratio")
done
if [ $new_conv_rank -ne 0 ]; then
    ext_args+=("--new_conv_rank=$new_conv_rank")
fi
# 运行svd_merge
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/networks/svd_merge_lora.py" \
    --save_precision="$save_precision" \
    --precision="$precision" \
    --new_rank="$new_rank" \
    --save_to="$save_to" \
    --device="$device" \
    "${ext_args[@]}"

echo "SVD合并完成"
read -rsp "按任意键继续..." -n1 key
