import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill
from .awe_entropy import dp_waypoint_selection, dp_entropy_waypoint_selection

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

    def speed_entropy(self,action, entropy, threshold):
        actions = []
        controller_mode = []
        # print("len_entropy",len(entropy))
        i = -1
        # actions.append(action[3,:])
        # actions.append(action[7,:])
        speeds = []
        max_speed = 3
        while (i+1)<len(entropy):
            if entropy[i+1]>threshold:
                for k in range(1, max_speed+1):
                    if entropy[min(k+i,len(entropy)-1)] <= threshold:
                        k = k-1
                        break
                high_speed = min(k, max_speed)
                index = min(i+high_speed, len(entropy)-1)
                actions.append(action[index,:])
                speed = high_speed
                i = i + speed
                speeds.append(speed)
                controller_mode.append(1)
            else:
                actions.append(action[i+1,:])
                speed = 3
                i = i+speed 
                speeds.append(speed)
                controller_mode.append(0)
        speeds = np.array(speeds)        
        actions = np.array(actions)
        actions = np.concatenate((actions, speeds[:,None]),axis=-1)
        # print(actions.shape)
        # 这里应该改成actions和controller一起输出这样就可以在4x部分加入controller了
        return actions

    def speed_awe_entropy(self,actions, entropy, threshold):
        controller_mode = []
        i = -1
        
        # waypoints = dp_waypoint_selection(actions=actions, err_threshold=0.005, pos_only=False)
        waypoints = dp_entropy_waypoint_selection(actions=actions, err_threshold=0.005, pos_only=False)
        actions = actions[waypoints]
        waypoints.insert(0,0)
        speeds = np.diff(np.array(waypoints))
        actions = np.concatenate((actions, speeds[:,None]),axis=-1)
        # print(actions.shape)
        # 这里应该改成actions和controller一起输出这样就可以在4x部分加入controller了
        return actions

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        # a_step = 0
        openloop = True
        entropy = action[:,-1]
        threshold = 0.002
        if openloop is True:
            # print("True")
            action = self.speed_awe_entropy(action, entropy, threshold)
        for k, act in zip(range(len(action)),action):
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
            if k<len(action)-1:
                if action[k, -2] > threshold and action[k+1, -2]<threshold:
                    eps = 0.02
                    diff = np.mean(np.abs(observation[14:17]-act[0:3]))
                    # done = self.controller_closeloop(act,eps,diff,done)
            # print(reward)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        if not self.done[-1] and entropy[-1]>threshold:
            eps = 0.02
            diff = np.mean(np.abs(observation[14:17]-act[0:3]))
            # done = self.controller_closeloop(act,eps,diff,done)

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
            if len(self.reward) > 0 and (self.reward[-1]==1):
                done = True
                self.done.append(done)
                break
            if idx > 1:
                break
            act[-1] = 0
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
        
        return done
