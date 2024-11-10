import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps
        self.statelist = env.statelist
        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs
    def speed_entropy(self,action):
        actions = []
        controller_mode = []
        entropy = action[:,-1]
        # print("len_entropy",len(entropy))
        # actions.append(action[1,:])
        # actions.append(action[3,:])
        # actions.append(action[5,:])
        # actions.append(action[7,:])
        # actions.append(action[3,:])
        # actions.append(action[7,:])
        # actions.append(action[2,:])
        # actions.append(action[5,:])
        # actions.append(action[7,:])
        
        # # 用点判断
        # i = -1
        # speed = 5
        # while (i+1)<len(entropy):
        #     if (i+speed)<len(entropy) and entropy[i+1]>0.04:
        #         actions.append(action[i+speed,:])
        #         i=i+speed
        #         controller_mode.append(1)
        #     else:
        #         actions.append(action[i+1,:])
        #         i = i+1 #+2
        #         controller_mode.append(0)
        # 用段判断
        i = -1
        speed = 4
        while (i+1)<len(entropy):
            if (i+speed)<len(entropy) and np.mean(entropy[i+1:i+speed])>0.04:
            # if (i+speed)<len(entropy) and entropy[i+1]>0.002:
                # print("a")
                actions.append(action[i+speed,:])
                i=i+speed
                controller_mode.append(1)
            else:
                actions.append(action[i+1,:])
                i = i+1 #+2
                controller_mode.append(0)
                
        actions = np.array(actions)
        # print(actions.shape)
        # 这里应该改成actions和controller一起输出这样就可以在4x部分加入controller了
        return actions,controller_mode
            
    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        # a_step = 0
        eps = 0.02
        openloop = True
        if openloop is True:
            # print("True")
            action,controller_mode = self.speed_entropy(action)
            # print("action",action.shape)
            # print("controller",controller_mode.shape)
        idx = 0
        for act in action:
            controller = controller_mode[idx]
            # a_step+=1
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            # print(self.reward)
            if len(self.reward) > 0 and (self.reward[-1]==1):
                done = True
                self.done.append(done)
                break
            observation, reward, done, info = super().step(act)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)
            # if controller==0:
            #     print("ori",np.mean(np.abs(observation[14:17]-act[0:3])))
            if controller==1:
                diff = np.mean(np.abs(observation[14:17]-act[0:3]))
                # print("speed",diff)
                done = self.controller_closeloop(act,eps,diff,done)
            idx+=1
            
        # print(a_step)
        # eps = 0 # 这里就是阈值
        # act = action[-1,:] # 这里就应该是结合controller_mode来的 但是为了应用action我就直接取每组最后一个了
        # done = self.controller_closeloop(act,eps,diff=2,done = aggregate(self.done, 'max'))
        observation = self._get_obs(self.n_obs_steps)
        # print("self.reward_agg_method",self.reward_agg_method)
        reward = aggregate(self.reward, self.reward_agg_method)
        # if reward == 1:
        #     print(reward)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    def controller_closeloop(self,act,eps,diff,done):
        idx = 0
        while done is False and diff > eps:
            if len(self.done) > 0 and self.done[-1]:
                break
            # print(self.reward)
            if len(self.reward) > 0 and (self.reward[-1]==1):
                done = True
                self.done.append(done)
                break
            if idx > 1:
                break
            observation, reward, done, info = super().step(act)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)
            idx+=1
            diff = np.mean(np.abs(observation[14:17]-act[0:3]))
            # diff = observation-act
        else:
            diff = 0   # 这里应该删除，但是现在diff 的rpy没有出来所以就直接给一个输出 
        return done
