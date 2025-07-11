import os
import os.path as osp
import datetime
import dateutil
import json

from acac.acac_marl.misc.logger import logger
from acac.acac_marl import PROJECT_DIR
# from wandb_utils import wandb_init


kst = datetime.timezone(datetime.timedelta(hours=9))
BASE_LOG_DIR = os.path.join(PROJECT_DIR, 'data', 'log')
BASE_SNAPSHOT_DIR = os.path.join(PROJECT_DIR, 'data', 'snapshot')

def create_exp_name(exp_prefix, env_name, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    timestamp = datetime.datetime.now(kst).strftime('%y%m%d_%H:%M:%S')
    return osp.join(exp_prefix, env_name, f's{seed}_{timestamp}')


def create_log_dir(
        exp_prefix,
        env_name,
        seed=0,
        base_log_dir=None,
):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param base_log_dir: The directory where all log should be saved.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, env_name, seed=seed)
    if base_log_dir is None:
        base_log_dir = BASE_LOG_DIR

    log_dir = osp.join(base_log_dir, exp_name)
    snapshot_dir = osp.join(BASE_SNAPSHOT_DIR, exp_name)

    for dir_ in [log_dir, snapshot_dir]:
        if osp.exists(dir_):
            print("WARNING: Log directory already exists {}".format(dir_))
        os.makedirs(dir_, exist_ok=True)

    return log_dir, snapshot_dir

def setup_logger(
        exp_prefix="default",
        variant=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        log_dir=None,
        script_name=None,
        **create_log_dir_kwargs
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

        based_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.

    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param variant:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param script_name: If set, save the script name to this.
    :return:
    """

    first_time = log_dir is None
    if first_time:
        log_dir, snapshot_dir = create_log_dir(exp_prefix, **create_log_dir_kwargs)
    else:
        snapshot_dir = log_dir

    # wandb_run = wandb_init(exp_prefix, log_dir, variant, **create_log_dir_kwargs)
    # logger.set_wandb_run(wandb_run)

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(tabular_log_path, logger._tabular_outputs,
                           logger._tabular_fds, mode='a')
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(snapshot_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)

    logger.log(f'scripts.logging_utils.py | setup_logger | log_dir: {log_dir}', with_prefix=False, with_timestamp=False)
    logger.log(f'scripts.logging_utils.py | setup_logger | snapshot_dir: {logger.get_snapshot_dir()}', with_prefix=False, with_timestamp=False)
    logger.log(f'scripts.logging_utils.py | setup_logger | snapshot_mode: {logger.get_snapshot_mode()}', with_prefix=False, with_timestamp=False)
    logger.log(f'scripts.logging_utils.py | setup_logger | snapshot_gap: {logger.get_snapshot_gap()}', with_prefix=False, with_timestamp=False)
    return log_dir


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d

def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False

