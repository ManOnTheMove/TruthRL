#!/bin/bash
#SBATCH --job-name=TruthRL-GRPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm-truthrl-%j.out
#SBATCH --error=slurm-truthrl-%j.err

# ============================================
# 路径配置
# ============================================
PROJECT_DIR=/home/erichyu/projects/def-zshakeri/erichyu
TRUTHRL_DIR=$PROJECT_DIR/embc/TruthRL
SIF_FILE=$PROJECT_DIR/envs/truthrl_apptainer/truthrl.sif
DATA_DIR=$TRUTHRL_DIR/data/TruthRL-CRAG
MODEL_DIR=$PROJECT_DIR/models/org

# Apptainer 临时目录
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/apptainer_cache
export APPTAINER_TMPDIR=$SLURM_TMPDIR/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# ============================================
# 加载 Apptainer
# ============================================
module load apptainer

# ============================================
# 训练参数（根据 GPU 数量调整）
# ============================================
N_GPUS=4                    # 根据实际 GPU 数量修改
ROLLOUT_TP_SIZE=2           # rollout tensor parallel size
BSZ=32                      # batch size，4xH100 建议 32
LR=1e-6
KL_LOSS_COEF=0.001

# 如果用本地模型路径（已下载），替换 HuggingFace 路径
# 注意：这里需要指向实际的模型路径
MODEL_NAME=$MODEL_DIR/meta-llama--Llama-3.1-8B-Instruct

# ============================================
# Verifier 配置
# ============================================
# 如果 verifier 在另一个作业中运行，修改下面的地址
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="token-abc123"

# ============================================
# WandB 配置
# ============================================
export WANDB_PROJECT="TruthRL"

# ============================================
# 运行训练
# ============================================
# 注意：
# 1. 确保 DATA_DIR 下有 train.parquet 和 test.parquet
# 2. 确保 verifier 服务已启动
# 3. --cleanenv 防止环境变量污染
# 4. -W $SLURM_TMPDIR 使用本地 SSD 作为工作目录 (这对 Ray 很重要)

apptainer exec --nv \
  --cleanenv \
  -B /project \
  -B /scratch \
  -B $HOME/.cache:$HOME/.cache \
  -W $SLURM_TMPDIR \
  --env "RAY_DEDUP_LOGS=0" \
  --env "OPENAI_API_BASE=$OPENAI_API_BASE" \
  --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --env "WANDB_PROJECT=$WANDB_PROJECT" \
  --env "PYTHONPATH=$TRUTHRL_DIR/training/verl" \
  --env "TOKENIZERS_PARALLELISM=false" \
  $SIF_FILE \
  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$BSZ \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name="TruthRL-Llama3.1-8B_bsz_${BSZ}_lr_${LR}_kl_${KL_LOSS_COEF}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.total_epochs=1
