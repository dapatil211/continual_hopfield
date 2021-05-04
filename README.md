# Hopfield Networks for Rehearsal Based Continual Learning

## Setting up Environment/Installing
This code was tested with Python 3.8.

To set up this repository, first clone it to a system with conda/miniconda. Then run the following commands:
```
cd continual_hopfield
git submodule init
git submodule update
conda env create --file env.yml
conda activate hopfield
```

That should install all the necessary requirements for this repository.

## Running
Make sure that you have [wandb](https://www.wandb.com/) logging set up on your machine.
The main script for the code is `main.py`. The usage is as follows:
```
usage: main.py [-h] [-m {tem,hopfield,dgr,finetune}] [-d {split_cifar100,split_mnist}] [--img-size IMG_SIZE [IMG_SIZE ...]] [--seed SEED] [--buffer-size BUFFER_SIZE] [--batch-size BATCH_SIZE] [--embed-dim EMBED_DIM] [--lr LR] [--beta BETA] [--replay-weight REPLAY_WEIGHT]
               [--hopfield-prob HOPFIELD_PROB] [--cross-validation] [--data-dir DATA_DIR] [--cifar-split CIFAR_SPLIT] [--mnist-split MNIST_SPLIT] [--output-file OUTPUT_FILE] [--run-name RUN_NAME] [--project-name PROJECT_NAME] [--same-head] [--wide-resnet] [--disable-cuda]
               [--learn-examples]

optional arguments:
  -h, --help            show this help message and exit
  -m {tem,hopfield,dgr,finetune}, --model-name {tem,hopfield,dgr,finetune}
  -d {split_cifar100,split_mnist}, --dataset-name {split_cifar100,split_mnist}
  --img-size IMG_SIZE [IMG_SIZE ...]
  --seed SEED
  --buffer-size BUFFER_SIZE
  --batch-size BATCH_SIZE
  --embed-dim EMBED_DIM
  --lr LR
  --beta BETA
  --replay-weight REPLAY_WEIGHT
  --hopfield-prob HOPFIELD_PROB
  --cross-validation
  --data-dir DATA_DIR
  --cifar-split CIFAR_SPLIT
  --mnist-split MNIST_SPLIT
  --output-file OUTPUT_FILE
  --run-name RUN_NAME
  --project-name PROJECT_NAME
  --same-head
  --wide-resnet
  --disable-cuda
  --learn-examples
  ```

  For an example of how to run, take a look at `run_exp.sh`.
  