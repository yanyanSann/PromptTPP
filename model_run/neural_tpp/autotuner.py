import glob
import os
import yaml
import subprocess
import time


def load_experiment_ids(config_dir):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    experiment_id_list = []
    for config in model_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            experiment_id_list += config_dict.keys()
    return sorted(experiment_id_list)


def grid_search(version, config_dir, gpu_list, expid_tag=None):
    experiment_id_list = load_experiment_ids(config_dir)
    if expid_tag is not None:
        experiment_id_list = [expid for expid in experiment_id_list if str(expid_tag) in expid]
        assert len(experiment_id_list) > 0, "tag={} does not match any expid!".format(expid_tag)
    gpu_list = list(gpu_list)
    idle_queue = list(range(len(gpu_list)))
    processes = dict()
    while len(experiment_id_list) > 0:
        if len(idle_queue) > 0:
            idle_idx = idle_queue.pop(0)
            gpu_id = gpu_list[idle_idx]
            expid = experiment_id_list.pop(0)
            cmd = "python -u run_expid.py --version {} --config {} --expid {} --gpu {}" \
                .format(version, config_dir, expid, gpu_id)
            # print("Run cmd:", cmd)
            p = subprocess.Popen(cmd.split())
            processes[idle_idx] = p
        else:
            time.sleep(5)
            for idle_idx, p in processes.items():
                if p.poll() is not None:  # terminated
                    idle_queue.append(idle_idx)
    [p.wait() for p in processes.values()]
