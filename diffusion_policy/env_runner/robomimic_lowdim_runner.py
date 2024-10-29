import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import os
import matplotlib.pyplot as plt
def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(512,512),
            render_camera_name='agentview',
            fps=20,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            speed=1,
            closeloop=False,
            te=False
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)
        render_hw=(512,512)
        fps = 20
        if n_envs is None:
            n_envs = n_train + n_test
        # import pdb
        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render,
                        pusht=False,
                        robomimic=True,
                        abs_action=abs_action,
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )
        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        # pdb.set_trace()
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)
        # pdb.set_trace()
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps #n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.outputdir = output_dir
        self.temporal_agg = te
        self.closeloop = closeloop

    def run(self, policy: BaseLowdimPolicy,speed = 1):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        max_entro = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            steps = np.zeros(n_envs)
            done = False
            t = 0
            state_dim = 7
            tasks_num = n_envs
            num_samples = 10
            # print(tasks_num)
            if self.temporal_agg:
                all_time_actions = torch.zeros(
                    [self.max_steps, tasks_num, self.max_steps + self.n_action_steps, state_dim]
            )   
                all_time_samples = torch.zeros(
                    [self.max_steps, tasks_num, self.max_steps + self.n_action_steps, num_samples,state_dim-1]
            )   

            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                if self.closeloop is True and self.temporal_agg is False:
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)
                else:        
                    with torch.no_grad():
                        action_dict = policy.fast_predict_action(obs_dict,speed = speed)
                        sample_dict = policy.get_samples(obs_dict, num_samples=num_samples,speed = speed)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
    
                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action_pred'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    # print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                
                
                # process samples:
                np_sample_dict = dict_apply(sample_dict,
                    lambda x: x.detach().to('cpu').numpy())

                sample = np_sample_dict['action_pred'][:,self.n_latency_steps:]
                if self.abs_action:
                    sample = self.undo_transform_action(sample)
                    
                sample = sample.reshape(num_samples,sample.shape[0]//num_samples,sample.shape[1],sample.shape[2])
               
                # perform temporal ensemble    
                if self.temporal_agg:
                    all_actions = torch.from_numpy(env_action).float()
                    all_samples = torch.from_numpy(sample).float()
                    all_samples = all_samples.permute(1,2,0,3)
                    # all_actions扩维度 最开始增加维度
                    all_actions = all_actions.unsqueeze(0)
                    all_time_actions[[t], :, t : t + self.n_action_steps] = all_actions  # [400, 56, 450, 14]
                    actions_for_curr_step = all_time_actions[:, :, t]  # [400, 56, 14]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=2)  # [400, 56]
                    all_time_samples[[t], :, t : t + self.n_action_steps] = all_samples[:,:,:,:6]  # [400, 56, 450, 10,14]
                    samples_for_curr_step = all_time_samples[:, :, t]  # [400, 56, 10,14]
                    
                    tasks_actions_for_curr_step = []
                    tasks_samples_for_curr_step = []
                    for task_idx in range(all_actions.shape[1]):  # 遍历任务
                        populated_actions = actions_for_curr_step[:, task_idx][actions_populated[:, task_idx]]
                        tasks_actions_for_curr_step.append(populated_actions)
                        populated_samples = samples_for_curr_step[:, task_idx][actions_populated[:, task_idx]]
                        tasks_samples_for_curr_step.append(populated_samples)
                    entropy = []
                    idx = 0
                    for task_samples in tasks_samples_for_curr_step:
                        entro = torch.mean(torch.var(task_samples.flatten(0,1),dim=0,keepdim=True),dim=-1,keepdim=True)
                        if max_entro[idx]<entro:
                            max_entro[idx] = entro
                        idx+=1
                        entropy.append(entro)
                    if max_entro is None:
                        max_entro = entropy
                    entropy = torch.stack(entropy)
                    entropy = entropy.detach().cpu().numpy()
                    
                    k = 0.01
                    weighted_actions = []
                    for task_actions in tasks_actions_for_curr_step:
                        exp_weights = np.exp(-k * np.arange(len(task_actions)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                        raw_action = (task_actions * exp_weights).sum(dim=0, keepdim=True)
                        weighted_actions.append(raw_action)
                  
                    weighted_actions = torch.stack(weighted_actions)  
                    env_action = weighted_actions.detach().cpu().numpy()
                    t+=1
                # if self.temporal_agg:
                #     all_actions = torch.from_numpy(env_action).float() #(28, 8, 7)
                #     all_samples = torch.from_numpy(sample).float()
                #     all_samples = all_samples.permute(1,2,0,3)
                #     # print("all_sample",all_samples.shape)
                #     # all_actions扩维度 最开始增加维度
                #     all_actions = all_actions.unsqueeze(0)
                #     all_time_actions[[t], :, t : t + self.n_action_steps] = all_actions  # [400, 56, 450, 14]
                #     actions_for_curr_step = all_time_actions[:, :, t]  # [400, 56, 14]
                #     actions_populated = torch.all(actions_for_curr_step != 0, axis=2)  # [400, 56]
                #     all_time_samples[[t], :, t : t + self.n_action_steps] = all_samples[:,:,:,:6]  # [400, 56, 450, 10, 14]
                #     samples_for_curr_step = all_time_samples[:, :, t]  # [400, 56, 10,14]
                #     tasks_actions_for_curr_step = []
                #     tasks_samples_for_curr_step = []
                #     for task_idx in range(all_actions.shape[1]):  # 遍历任务
                #         populated_actions = actions_for_curr_step[:, task_idx][actions_populated[:, task_idx]]
                #         tasks_actions_for_curr_step.append(populated_actions)
                #         populated_samples = samples_for_curr_step[:, task_idx][actions_populated[:, task_idx]]
                #         tasks_samples_for_curr_step.append(populated_samples)
                #     tasks_samples_for_curr_step = torch.stack(tasks_samples_for_curr_step) if isinstance(tasks_samples_for_curr_step, list) else tasks_samples_for_curr_step
                #     # print(tasks_samples_for_curr_step.shape)
                #     # 计算所有任务的熵
                #     task_variances = torch.var(tasks_samples_for_curr_step.flatten(1, 2), dim=1, keepdim=True)  # [num_tasks, 1, num_features]
                #     entropy = torch.mean(task_variances, dim=-1, keepdim=True)  # [num_tasks, 1, 1]

                #     # 如果 max_entro 还未初始化，直接将 entropy 赋值给它
                #     if max_entro is None:
                #         max_entro = entropy.clone()
                #     else:
                #         # 更新 max_entro 中每个任务的熵值，只保留较大的值
                #         max_entro = torch.max(max_entro, entropy)

                #     # 检查 entropy 的形状
                #     # print(entropy.shape)  # 应为 (num_tasks, 1, 1)

                #     entropy = entropy.detach().cpu().numpy()
                #     tasks_actions_for_curr_step = torch.stack(tasks_actions_for_curr_step) if isinstance(tasks_actions_for_curr_step, list) else tasks_actions_for_curr_step

                #     k = 0.01
                #     # 获取任务数量和动作序列长度
                #     # 检查 tasks_actions_for_curr_step 形状
                    

                #     # 检查 exp_weights 形状
                #     num_tasks, num_actions = tasks_actions_for_curr_step.shape[:2]

                #     # 计算所有动作的指数加权系数，并扩展维度以便与 tasks_actions_for_curr_step 对齐
                #     exp_weights = torch.from_numpy(np.exp(-k * np.arange(num_actions))).unsqueeze(0).expand(num_tasks, -1)  # [1, num_actions]
                #     exp_weights = exp_weights / exp_weights.sum()  # 归一化权重
                #     exp_weights = exp_weights.to(tasks_actions_for_curr_step.device)  # 确保在同一设备上
                #     exp_weights = exp_weights.unsqueeze(-1)
                #     # print("tasks_actions_for_curr_step shape:", tasks_actions_for_curr_step.shape) 
                #     # print("exp_weights shape before view:", exp_weights.shape)
                #     # 广播乘法以对每个任务的所有动作应用权重，然后对动作维度求和
                #     weighted_actions = (tasks_actions_for_curr_step * exp_weights).sum(dim=1, keepdim=True)  # [batch, num_tasks, features]

                #     # weighted_actions = torch.stack(weighted_actions)  
                #     env_action = weighted_actions.detach().cpu().numpy()
                #     print(env_action.shape)
                #     import pdb;pdb.set_trace()
                #     t+=1


                # print(entropy.shape)
                env_action = np.concatenate((env_action, entropy),axis=-1)
                obs, reward, done, info = env.step(env_action)
                steps += (reward == 0)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(env_action.shape[1])
            pbar.close()
            step_file_path = os.path.join(self.outputdir, 'step.txt')
            with open(step_file_path, 'a') as f:
                f.write("Steps:\n")
                f.write(", ".join([str(step) for step in steps]))  # Writing the steps as a comma-separated list
                f.write("\n")
                f.write("max-entropy:\n")
                f.write(", ".join([str(entro) for entro in max_entro]))
                f.write("\n")

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        max_rewards = collections.defaultdict(list)
        log_data = dict()
        self.plot_action_vs_pos_agent(save_dir = self.outputdir , env = env)
        
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        # print(d_rot)
        pos = action[...,:3]
        # print(pos[0,0,:])
        rot = action[...,3:3+d_rot]
        # print(rot[0][0][:])
        gripper = action[...,[-1]]
        # print(gripper[0,0,:])
        rot = self.rotation_transformer.inverse(rot)
        # print(rot[0,0,:])
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    def plot_action_vs_pos_agent(self, save_dir, env):
        # 遍历每个任务
        for i in range(len(env.statelist)):
            task_data = env.statelist[i][0]  # 获取每个任务的数据

            # 初始化存储 7 维度的 action 和 pos_agent
            actions = [[] for _ in range(7)]
            pos_agent = [[] for _ in range(7)]

            # 遍历任务中的每一步
            for step in task_data:
                if isinstance(step, list):
                    step = step[0]
                action = step['action']
                pos = step['pos_agent']

                # 将每个维度的 action 和 pos_agent 存入对应列表
                for j in range(7):
                    actions[j].append(action[j])
                    pos_agent[j].append(pos[j])

            # 创建时间步
            tstep = np.linspace(0, 1, len(actions[0]) - 1)
            n_groups = 7  # 7 个 group, 用于每个维度

            # 创建子图
            fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
            save_path = os.path.join(save_dir, 'plot', f'rollout{i+1}_action_vs_pos_agent.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 绘制每个维度的对比
            for j in range(7):
                axes[j].plot(tstep, np.array(actions[j])[1:], label=f'action_dim_{j}')
                axes[j].plot(tstep, np.array(pos_agent[j])[1:], label=f'pos_agent_dim_{j}')
                axes[j].set_title(f'Task {i+1} Dimension {j}: Action vs Pos Agent')
                axes[j].legend()

            plt.xlabel('Timestep')
            plt.tight_layout()

            # 保存图表
            fig.savefig(save_path)
            plt.close(fig)

