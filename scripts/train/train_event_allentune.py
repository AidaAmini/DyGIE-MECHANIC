# USAGE: `bash train.sh [config_name]`
#
# The `config_name` is the name of one of the `jsonnet` config files in the
# `training_config` directory, for instance `scierc`. The result of training
# will be placed under `models/[config_name]`.




import argparse
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict
from pathlib import Path
import pathlib

"""
Usage:
python scripts/train/train_event_allentune.py --data_dir data/processed/collated_events/ --serial_dir ./models/events --gpu_count 4 --device 0,1,2,3
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name

    parser.add_argument('--config',
                        type=str,
                        default="./training_config/covid-t.jsonnet",
                        help='path to training config file',
                        required=False)

    parser.add_argument('--search_space',
                        type=str,
                        default="./training_config/search_space.json",
                        help='path to the search parameter config',
                        required=False)

    parser.add_argument('--data_dir',
                        type=Path,
                        help='root dataset folder, contains train,dev,test',
                        required=True,
                        default="data/processed/collated/")
    
    parser.add_argument('--train_file',
                            type=str,
                            help="Please mention train filename in the data_dir if train filename is not train.json",
                            required=False,
                            default="train.json")

    parser.add_argument('--dev_file',
                            type=str,
                            help="Please mention dev filename in the data_dir if dev filename is not dev.json",
                            required=False,
                            default="dev.json")

    parser.add_argument('--test_file',
                            type=str,
                            help="Please mention test filename in the data_dir if test filename is not test.json",
                            required=False,
                            default="test.json")

    parser.add_argument('--serial_dir',
                            type=str,
                            help="Path to the directory to save the model. default is ./models/",
                            required=False,
                            default="./models/")

    parser.add_argument('--device',
                        type=str,
                        default='1,2,3',
                        required=False,
                        help="cuda devices comma seperated")

    parser.add_argument('--gpu_count',
                        type=int,
                        default=3,
                        required=False,
                        help="number of GPUs to be used in the experiment")

    parser.add_argument('--cpu_count',
                        type=int,
                        default=32,
                        required=False,
                        help="number of GPUs to be used in the experiment")

    parser.add_argument('--master_port',
                        type=str,
                        default="2424",
                        help='for pytorch distributed training',
                        required=False)

    parser.add_argument('--num_samples',
                        type=int,
                        default=30,
                        required=False,
                        help="how many samples of trials should be run. default is 30.")

    args = parser.parse_args()
    data_root = pathlib.Path(args.data_dir) 
    serial_dir = pathlib.Path(args.serial_dir)
    config_file = args.config

    print("saving model to : " + str(serial_dir))
    ie_train_data_path = data_root/ args.train_file
    ie_dev_data_path = data_root/ args.dev_file
    ie_test_data_path = data_root/ args.test_file
    os.environ['ie_train_data_path'] = str(ie_train_data_path)
    os.environ['ie_dev_data_path'] = str(ie_dev_data_path)
    os.environ['ie_test_data_path'] = str(ie_test_data_path)


    if args.device:
        os.environ['CUDA_DEVICE'] = args.device
        os.environ['cuda_device'] = args.device
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        os.environ['master_port'] = args.master_port

    allennlp_command = [
            "allentune",
            "search",
            "--experiment-name",
            str(serial_dir),
            "--num-gpus",
            str(args.gpu_count),
            "--num-cpus",
            str(args.cpu_count),
            "--gpus-per-trial",
            "1",
            "--cpus-per-trial",
            "1",  
            "--search-space",
            args.search_space,
            "--num-samples",
            str(args.num_samples),
            "--base-config",
            str(args.config),
            "--include-package",
            "dygie"
    ]
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)







