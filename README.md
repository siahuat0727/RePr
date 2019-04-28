# RePr
My implementation of RePr training scheme in PyTorch. https://arxiv.org/pdf/1811.07275.pdf

## Usage
```
$ python main.py --help
usage: main.py [-h] [--lr LR] [--repr] [--S1 S1] [--S2 S2] [--epochs EPOCHS]
               [--workers WORKERS] [--print_freq PRINT_FREQ] [--gpu GPU]
               [--save_model SAVE_MODEL] [--prune_ratio PRUNE_RATIO]
               [--comment COMMENT] [--zero_init]

PyTorch CIFAR10 Training

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate (default: 0.01)
  --repr                whether to use RePr training scheme (default: False)
  --S1 S1               S1 epochs for RePr (default: 20)
  --S2 S2               S2 epochs for RePr (default: 10)
  --epochs EPOCHS       total epochs for training (default: 100)
  --workers WORKERS     number of worker to load data (default: 16)
  --print_freq PRINT_FREQ
                        print frequency (default: 50)
  --gpu GPU             gpu id (default: 0)
  --save_model SAVE_MODEL
                        path to save model (default: best.pt)
  --prune_ratio PRUNE_RATIO
                        prune ratio (default: 0.3)
  --comment COMMENT     tag for tensorboardX event name (default: )
  --zero_init           whether to initialize with zero (default: False)
```

## Execute example
Standard training scheme
```
$ python main.py
```

RePr training scheme
```
$ python main.py --repr --S1 20 --S2 10 --epoch 110
```

## Results

### Original paper

Std  | RePr
---- | ----
72.1 | 76.4

![](/img/paper-result.png)

### My implementation

With data augmentation (`torchvision.transforms.RandomCrop`)
learning rate = 0.01

Std  | RePr
---- | ----
77.84| 74.48

![](/img/lr0.01.png)


Without data augmentation
learning rate = 0.003

Std  | RePr
---- | ----
64.86| 69.05

![](/img/lr0.01.png)

For more information, please visit [my blog](https://siahuat0727.github.io/2019/03/17/repr/) (Last updated on 2019-04-28)
