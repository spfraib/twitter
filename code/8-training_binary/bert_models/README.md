# Training

## Preliminary steps:

- Ideally, create a virtual environment specifically for training and activate it:

```
$ python3 -m virtualenv env_name
$ source env_name/bin/activate
```

- Install the necessary packages:
```
$ cd twitter/code/8-training_binary/simpletransformers #in case you haven't cd into the present folder yet
$ pip install -r requirements.txt
```

- Install PyTorch separately without cache to not use too much memory:
`$ pip install --no-cache-dir torch==1.5.0`

- Install [apex](https://github.com/nvidia/apex) to be able to use fp16 training. On Linux, it is done this way:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

We are now ready to start training.

