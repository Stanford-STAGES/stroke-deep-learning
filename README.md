# stroke-deep-learning

# Sherlock commands.

#### run interactive gpu:
srun -p gpu --gres gpu:1 --pty bash

#### tensorboard:
```
[zyzhang@sherlock-ln03 login_node ~]$ salloc -N 1 -n 1 --time=48:00:00 --gres=gpu:1 -p gpu,hns_gpu
[zyzhang@sherlock-ln03 login_node ~]$ squeue -u zyzhang

JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)

17689718 gpu bash zyzhang R 25:24 1 gpu-17-35

[zyzhang@sherlock-ln03 login_node ~]$ ssh -XY gpu-17-35

[zyzhang@gpu-17-35 /scratch/users/zyzhang/usertests/tensorflow/distributed]$ ml load tensorflow.1/1.3.0

[zyzhang@gpu-17-35 /scratch/users/zyzhang/usertests/tensorflow/distributed]$ tensorboard --logdir=/scratch/users/zyzhang/usertests/tensorflow/logdir

Starting TensorBoard 55 at http://gpu-17-35.local:6006
```
(Press CTRL+C to quit)

I am on a local linux machine and then I can display the tensorboard with the following:
```
[zyzhang@srn-exciton ~]$ ssh -L 8000:gpu-17-35:6006 zyzhang@sherlock.stanford.edu
```
after the above, you can display the tensorboard in a web browser with http://localhost:8000/
