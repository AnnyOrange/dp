"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import shutil
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:4')
@click.option('-s', '--speed', default=1)
@click.option('-cl', '--closeloop', default=False)
@click.option('-t', '--te', default=False)
@click.option('-i', '--is_entropy', default=False)
def main(checkpoint, output_dir, device, speed,closeloop,te,is_entropy):
    if os.path.exists(output_dir):
        # click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        shutil.rmtree(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    if closeloop is True and te is False:
        cfg.task.env_runner['n_action_steps'] = speed
        policy.n_action_steps = speed
    else:
        cfg.task.env_runner['n_action_steps'] = int(cfg.task.env_runner['n_action_steps']//speed)
        policy.n_action_steps = int(policy.n_action_steps//speed)
        if speed==3:
            cfg.task.env_runner['n_action_steps'] = 3
            policy.n_action_steps = 3
    if te is True and closeloop is False:
        raise ValueError("Error: `te` is True and `closeloop` is False, which is not allowed.")
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        speed = speed,
        closeloop = closeloop,
        te = te,
        is_entropy = is_entropy)
    runner_log = env_runner.run(policy,speed = speed)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
