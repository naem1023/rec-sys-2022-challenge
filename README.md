# rec-sys-2022-challenge

[Challenge Link](http://www.recsyschallenge.com/2022/)

## Baseline

## Download data
```sh
chmod 755 download.sh
./download.sh
```

## Environment
1. Run Nvidia Merlin container
```sh
chmod 755 run_merlin_container.sh
./run_merlin_container.sh
```
2. Run Jupyter Lab in nvidia-docker
```sh
cd /transformers4rec/examples
jupyter-lab --allow-root --ip='0.0.0.0' --port 8888
```

## Member

[Ho Jin Lee](https://github.com/ili0820), [Sungho Park](https://github.com/naem1023)