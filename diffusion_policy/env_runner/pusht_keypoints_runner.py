import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

import os
import matplotlib.pyplot as plt

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            speed=1,
            closeloop=False
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test
        max_steps = max_steps//speed
        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
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
                    pusht=True,
                    robomimic=False,
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
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
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

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        # print(n_obs_steps)
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.outputdir = output_dir
        self.temporal_agg = closeloop
    
    def run(self, policy: BaseLowdimPolicy,speed = 1):
        device = policy.device
        dtype = policy.dtype

        env = self.env
        
        # if temporal_agg:
            # query_frequency = 1
            # num_queries = policy_config["num_queries"]

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        # max_timesteps = 1000
        
        
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

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            state_dim = 2
            # num_queries = 8
            tasks_num = n_envs
            # print(tasks_num)
            if self.temporal_agg:
                all_time_actions = torch.zeros(
                    [self.max_steps, tasks_num, self.max_steps + self.n_action_steps, state_dim]
            )
            done = False
            t = 0
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                # self.past_action = True
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # print(obs_dict['obs'].shape)
                # import pdb;pdb.set_trace()
                # run policy
                with torch.no_grad():
                    action_dict = policy.fast_predict_action(obs_dict,speed = speed)
                
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'][:,self.n_latency_steps:]
                
                if self.temporal_agg:
                    all_actions = torch.from_numpy(action).float()
                    # all_actions扩维度 最开始增加维度
                    all_actions = all_actions.unsqueeze(0)
                    all_time_actions[[t], :, t : t + self.n_action_steps] = all_actions  # [400, 56, 450, 14]
                    # print(all_time_actions.shape)
                    actions_for_curr_step = all_time_actions[:, :, t]  # [400, 56, 14]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=2)  # [400, 56]
                    # print(actions_populated.shape)
                    # 使用 boolean indexing 保留 populated actions
                    tasks_actions_for_curr_step = []
                    for task_idx in range(all_actions.shape[1]):  # 遍历任务
                        populated_actions = actions_for_curr_step[:, task_idx][actions_populated[:, task_idx]]
                        # print(populated_actions.shape)
                        tasks_actions_for_curr_step.append(populated_actions)
                    # import pdb;pdb.set_trace()
                    # print(len(tasks_actions_for_curr_step))
                    k = 0.01
                    weighted_actions = []
                    for task_actions in tasks_actions_for_curr_step:
                        exp_weights = np.exp(-k * np.arange(len(task_actions)))
                        # print(exp_weights.shape)
                        # print(exp_weights)
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                        # 计算加权动作
                        raw_action = (task_actions * exp_weights).sum(dim=0, keepdim=True)
                        weighted_actions.append(raw_action)
                    # import pdb;pdb.set_trace()
                    # print(len(weighted_actions))
                    # 最终的 weighted_actions 是包含 56 个 [1, 14] 的列表
                    weighted_actions = torch.stack(weighted_actions)  # 将所有任务的加权动作堆叠成 tensor
                    action = weighted_actions.detach().cpu().numpy()
                    # print(action.shape)
                    # import pdb;pdb.set_trace()
                    t+=1
                    # print(action.shape)
                obs, reward, done, info = env.step(action)
                # print('info',info)
                # import pdb;pdb.set_trace()
                done = np.all(done)
                past_action = action
                # idx+=1

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            # print("yeah")
        # import pdb; pdb.set_trace()
        # print(env.statelist)
        # first_state = env.statelist[0]
        # # print(first_state)
        # print(len(first_state)) # 1
        # print(len(first_state[0])) # 169
        # print(first_state[0][0]) # [{'action': array([251.74745, 113.89054], dtype=float32), 'pos_agent': array([230.79548219, 101.99405645])}]
        # print(len(first_state[0][0])) # 1
        # print(len(env.statelist)) # 56
        # import pdb;pdb.set_trace()
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # 画图累
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
    def set_sampler(self, sampler, nsample=1, nmode=1, noise=0.0, decay=1.0):
        self.sampler = sampler
        self.n_samples = nsample
        self.nmode = nmode
        self.noise = noise
        self.decay = decay
        if noise > 0:
            self.disruptor = NoiseGenerator(self.
                noise)
        print(f'Set sampler: {sampler} {nsample}/{nmode}')

    def set_reference(self, weak):
        self.weak = weak
    

    def plot_action_vs_pos_agent(self, save_dir, env):
        # 遍历每个任务
        for i in range(len(env.statelist)):
            task_data = env.statelist[i][0]  # 获取每个任务的数据
            print(len(task_data))
            # 初始化存储 X 和 Y 方向的 action 和 pos_agent
            actions_x = []
            actions_y = []
            pos_agent_x = []
            pos_agent_y = []

            # 遍历任务中的每一步
            for step in task_data:
                if isinstance(step, list):
                    step = step[0]
                action = step['action']
                pos_agent = step['pos_agent']

                # 提取 action 和 pos_agent 的 x 和 y 值
                actions_x.append(action[0])
                actions_y.append(action[1])
                pos_agent_x.append(pos_agent[0])
                pos_agent_y.append(pos_agent[1])

            # 创建时间步
            tstep = np.linspace(0, 1, len(actions_x) - 1)
            n_groups = 2  # 2 个 group, 一个用于 X 方向，一个用于 Y 方向

            # 创建子图
            fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True)
            save_path = os.path.join(save_dir, 'plot', f'rollout{i+1}_action_vs_pos_agent.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 绘制 X 方向的对比
            axes[0].plot(tstep, np.array(actions_x)[1:], label=f'action_x')
            axes[0].plot(tstep, np.array(pos_agent_x)[1:], label=f'pos_agent_x')
            axes[0].set_title(f'Task {i+1} X axis: Action vs Pos Agent')
            axes[0].legend()

            # 绘制 Y 方向的对比
            axes[1].plot(tstep, np.array(actions_y)[1:], label=f'action_y')
            axes[1].plot(tstep, np.array(pos_agent_y)[1:], label=f'pos_agent_y')
            axes[1].set_title(f'Task {i+1} Y axis: Action vs Pos Agent')
            axes[1].legend()

            plt.xlabel('Timestep')
            plt.tight_layout()

            # 保存图表
            fig.savefig(save_path)
            plt.close(fig)
