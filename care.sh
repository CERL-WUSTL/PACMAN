
source ~/.bashrc

conda activate FairRL





export HYDRA_FULL_ERROR=1
export num_envs=50
PYTHONPATH=. python -u ../main.py \
setup=metaworld \
setup.algo=care_mt${num_envs} \
env=metaworld-mt${num_envs} \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=1500000 \
experiment.eval_only=False \
experiment.random_pos=False \
setup.seed=3 \
experiment.few_shot=True \
setup.base_path=<base_path>\
setup.dir_name=<dir_path>\
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=${num_envs} \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=True \
agent.encoder.type_to_select=moe \
agent.multitask.should_use_multi_head_policy=False \
agent.encoder.moe.task_id_to_encoder_id_cfg.mode=attention \
agent.encoder.moe.num_experts=6 \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True \
agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=True \
agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.path_to_load_from=/CMTA/metadata/task_embedding/roberta_small/metaworld-mt50.json \

