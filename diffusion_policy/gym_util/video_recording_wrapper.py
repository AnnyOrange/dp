import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            pusht=False,
            robomimic=False,
            abs_action=False,
            steps_per_render=1,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.statelist = []
        self.pusht = pusht
        self.robomimic = robomimic
        self.step_count = 0
        self.abs_action = abs_action

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        # print("step")
        # import pdb;pdb.set_trace()
        result = super().step(action)
        # print(result)
        # print(result)
        # print("obs",result[0])
        # print("action_in",action.shape)
        # robo_pos_
        # print(result)
        # 如果是pusht的画用这里的
        if self.pusht is True:
            state_data = [{
                "action": action,
                "pos_agent": result[-1]['pos_agent']
            }]
            self.statelist.append(state_data)
        if self.robomimic is True:
            # print(result[-1]['robot0_eef_pos'])
            robot0_eef_pos = result[-1]['robot0_eef_pos']
            robot0_eef_quat = result[-1]['robot0_eef_quat']
            print("robot0_eef_quat",len(robot0_eef_quat))
            robot0_gripper_qpos = result[-1]['robot0_gripper_qpos']
            print("robot0_gripper_qpos",len(robot0_gripper_qpos))
            agent = np.concatenate([robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos])
            # print(agent)
            if self.abs_action:
                rotation_transformer = RotationTransformer('quaternion', 'rotation_6d')
                agent = self.undo_transform_action(agent,rotation_transformer)
                print(agent)
                print(agent.shape)
            state_data = [{
                "action": action,
                "agent": agent
            }]
        result[-1] = {}
        print(result)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
    def undo_transform_action(self, agent,rotation_transformer):
        raw_shape = agent.shape
        print(raw_shape)
        if raw_shape[-1] == 20:
            # dual arm
            action = agent.reshape(2,10)

        d_rot = agent.shape[-1] - 5
        pos = agent[...,:3]
        rot = agent[...,3:3+d_rot]
        gripper = agent[...,[-1]]
        rot = rotation_transformer.inverse(rot)
        uagent = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uagent = uagent.reshape(*raw_shape[:-1], 14)

        return uagent

