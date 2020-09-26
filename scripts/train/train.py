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
python scripts/train/train.py --data_path data/cofie --device 0,1,2,3
python scripts/train/train.py --data_path data/cofie --device 0,1,2,3 --serial_dir models/cofie/


"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name

    parser.add_argument('--config',
                        type=str,
                        default="./training_config/scierc_lightweight.jsonnet",
                        help='path to training config file',
                        required=False)
 
    parser.add_argument('--data_path',
                        type=Path,
                        help='root dataset folder, contains train,dev,test',
                        required=True)
    
    parser.add_argument('--train_file',
                            type=str,
                            help="Please mention train filename in the data_path if train filename is not train.json",
                            required=False,
                            default="train.json")

    parser.add_argument('--dev_file',
                            type=str,
                            help="Please mention dev filename in the data_path if dev filename is not dev.json",
                            required=False,
                            default="dev.json")

    parser.add_argument('--test_file',
                            type=str,
                            help="Please mention test filename in the data_path if test filename is not test.json",
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

    parser.add_argument('--master_port',
                        type=str,
                        default="2424",
                        help='for pytorch distributed training',
                        required=False)

    args = parser.parse_args()
    data_root = pathlib.Path(args.data_path) 
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
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        os.environ['cuda_device'] = args.device
        os.environ['master_port'] = args.master_port

    allennlp_command = [
            "allennlp",
            "train",
            config_file,
            "--serialization-dir",
            str(serial_dir),
            "--include-package",
            "dygie"
    ]
    
   
    subprocess.run(" ".join(allennlp_command), shell=True, check=True)

