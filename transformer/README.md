



# Transformer
develop a seq2seq model based on the transformer architecture as described in the paper "Attention Is All You Need." The model will focus on a simple yet effective task: echoing user input. This project aims to demonstrate the capabilities of transformer models in processing and replicating sequences.

implement 2 types transformer, one is mainly borrowed from heavily borrowed from http://nlp.seas.harvard.edu/annotated-transformer/, one is coding from scratch. use '-t' option to choose .





## train

```
python demo_char_echo_model.py -N 1 --d_model 64 --d_ff 128 --batch_size 1024 --nbatches 2048 --epochs 1  -t anno --model_save_path out/echo_net.pt  --mode train
```


## test

```
python demo_char_echo_model.py -N 1 --d_model 64 --d_ff 128 --batch_size 1024 --nbatches 2048 --epochs 1  -t anno --model_save_path out/echo_net.pt  --mode test
```


## input
this mode reads user input. 

```
python demo_char_echo_model.py -N 1 --d_model 64 --d_ff 128 --batch_size 1024 --nbatches 2048 --epochs 1  -t anno --model_save_path out/echo_net.pt  --mode input
```
