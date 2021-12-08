#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

type=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['type'])")
arch=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['arch'])")
dataset=$(python -c "import yaml;print(yaml.load(open('${config}'))['data']['dataset'])")
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p exp/${type}/${arch}/${dataset}/${now}
srun -N 1 --cpus-per-task=12 -p huzhou-2 --gres=gpu:2  --qos high --job-name test -t 7-00:00:00 python -u train_swin.py --config ${config} --log_time $now 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log
# srun -N 1 --cpus-per-task=8 -p huzhou-2 --gres=gpu:2  --qos high --job-name test -t 7-00:00:00 python -u train.py --config ${config} --log_time $now 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log
# --mail-user=mengmengwang@zju.edu.cn --mail-type=ALL -x node86 
#_test_clip_crop