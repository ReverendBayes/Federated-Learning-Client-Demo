# Federated Learning Client Demo

## What it is 
A one-file FedAvg client trainer (e.g. on MNIST) that you can point at a server and immediately join a federated round.

## Why you need it
* Federated learning is exploding in edge/IoT and healthcare—but demos live in bulky frameworks (Flower, TFF).
* A single flclient.py would let researchers quickly prototype ideas without cloning ten repos.

## Key features
python flclient.py \
  --data mnist \
  --model cnn \
  --server http://localhost:8080
* Built-in model & dataset loaders
* Configurable round size, epochs per round
* Outputs local metrics JSON

## Run it with:
python flclient.py \
  --data mnist \
  --model cnn \
  --server http://localhost:8080 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 32 \
  --lr 0.01 \
  --metrics-out metrics.json

## Dependencies:
pip install torch torchvision requests
