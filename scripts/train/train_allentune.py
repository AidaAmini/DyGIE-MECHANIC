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
Usage
python scripts/train/train_allentune.py --data_dir data/cofie --device 0,1,2,3
python scripts/train/train_allentune.py --data_dir data/cofie --device 0,1,2,3 --serial_dir models/cofie/


"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name

    parser.add_argument('--config',
                        type=str,
                        default="./training_config/covid.jsonnet",
                        help='training config',
                        required=False)

    parser.add_argument('--search_space',
                        type=str,
                        default="./training_config/search_space.json",
                        help='training search space',
                        required=False)
 

    parser.add_argument('--data_dir',
                        type=Path,
                        help='root dataset folder, contains train,dev,test',
                        required=True)
    
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
    
    parser.add_argument('--num_samples',
                        type=int,
                        default=30,
                        required=False,
                        help="how many samples of trials should be run. default is 30.")

    parser.add_argument('--device',
                        type=str,
                        default='1,2,3',
                        required=False,
                        help="cuda devices comma seperated")


    args = parser.parse_args()

    data_root = pathlib.Path(args.data_dir) 
    serial_dir = pathlib.Path(args.serial_dir)
    config_file = args.config

    gpu_count = args.gpu_count
    cpu_count = args.cpu_count
    num_samples = args.num_samples
    config_file = args.config
    search_space = args.search_space
    

    print("saving model to : " + str(serial_dir))
    ie_train_data_path = data_root/ args.train_file
    ie_dev_data_path = data_root/ args.dev_file
    ie_test_data_path = data_root/ args.test_file
    os.environ['ie_train_data_path'] = str(ie_train_data_path)
    os.environ['ie_dev_data_path'] = str(ie_dev_data_path)
    os.environ['ie_test_data_path'] = str(ie_test_data_path)

    if args.gpu_count > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        os.environ['cuda_device'] = args.device
        os.environ['master_port'] = '2323'

    allennlp_command = [
            "allentune",
            "search",
            "--experiment-name",
            str(serial_dir),
            "--num-gpus",
            str(gpu_count),
            "--num-cpus",
            str(cpu_count),
            "--gpus-per-trial",
            str(1),
            "--cpus-per-trial",
            str(1),  
            "--search-space",
            search_space,
            "--num-samples",
            str(num_samples),
            "--base-config",
            config_file,
            "--include-package",
            "dygie"
    ]
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)







