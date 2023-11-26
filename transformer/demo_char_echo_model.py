
import os
import time
import argparse
from datetime import datetime


import matplotlib as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from char_gen_dataloader import *
from simple_loss import *
from seq2seq_model import *


# example_learning_schedule()

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='A simple program that echo your input chars.')
parser.add_argument('-t', '--transformer_type', type=str, required=True, help='the transformer type, anno or self.')
parser.add_argument('-N', '--num_blocks', type=int, required=True, help='the transformer layers num.')
parser.add_argument('--d_model', type=int, default=512, help='num epochs')
parser.add_argument('--d_ff', type=int, default=2048, help='num epochs')
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--nbatches', type=int, required=True, help='num batches')
parser.add_argument('--epochs', type=int, default=20, help='num epochs')
parser.add_argument('--model_save_path', type=str, default='simple_copy_model.pt', help='num epochs')
parser.add_argument('--mode', type=str, default='train', help='train, test, input')
args = parser.parse_args()

print(f'cuda available {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_len = 32
char_vocab = ExtendedVocab()

shape_option = 1   # batch mask shape option 

model = None
if args.transformer_type == 'self':  # self transformer
    # shape_option = 1  # D=4
    model = SeqModelSelf(len(char_vocab), len(char_vocab), N=args.num_blocks, d_model=args.d_model, d_ff=args.d_ff)
else:   # annotated transformer
    # shape_option = 0  # D=3
    model = SeqModelAnno(len(char_vocab), len(char_vocab), N=args.num_blocks, d_model=args.d_model, d_ff=args.d_ff)
    

def count_train_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    device,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_mask = batch.tgt_mask.to(device)
        tgt_y = batch.tgt_y.to(device)

        out = model.forward(
            # batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            src, tgt, src_mask, tgt_mask
        )
        # print(f'src : {src.shape} \n {src}')
        # print(f'tgt : {tgt.shape} \n {tgt}  \n out:{out.shape} \n {out}')
        loss, loss_node = loss_compute(out, tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print((
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e")
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr))
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []
    steps = 20000
    # steps = 20

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(steps):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    x = range(steps)
    y0 = learning_rates[0]
    y1 = learning_rates[1]
    y2 = learning_rates[2]

    plt.plot(x, y0, label='Line 1', color='blue', linestyle='-')
    plt.plot(x, y1, label='Line 2', color='red', linestyle='--')
    plt.plot(x, y2, label='Line 3', color='green', linestyle='-.')
    plt.savefig('lr_plot.png', dpi=300)
    plt.close()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def train(model, device,
          batch_size=80, nbatches=20, epochs=20,
          save_model_path='simple_copy_model.pt'):
    criterion = LabelSmoothing(size=len(char_vocab), padding_idx=char_vocab.pad_idx, smoothing=0.0)
    model = model.to(device=device)
    num_total_params = count_total_parameters(model)
    print('model num total params:', num_total_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=args.d_model, factor=1.0, warmup=400
        ),
    )

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = f"logs/{now}"
    writer = SummaryWriter(log_dir=logs_dir)
    params_desc = f'num blocks:{args.num_blocks}, d_model:{args.d_model}, d_ff:{args.d_ff}, batch size:{batch_size}, nbatches:{nbatches}, epochs:{epochs}'
    writer.add_text('params', params_desc)

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print('train epoch {}'.format(epoch))
        model.train()
        loss, _ = run_epoch(
            data_gen_extended_vocab(char_vocab, batch_size, nbatches, max_seq_len=max_len, shape_option=shape_option),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            device=device,
            mode="train",
        )
        print('train epoch {} , loss {}, '.format(epoch, loss))
        train_loss.append(loss.item())
        writer.add_scalar('train_loss', loss.item(), epoch)

        model.eval()
        loss = run_epoch(
            data_gen_extended_vocab(char_vocab, batch_size, nbatches, max_seq_len=max_len, shape_option=shape_option),
            model,
            SimpleLossCompute(model.generator, criterion),
            # SimpleLossCompute(model.output_layer, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            device=device,
            mode="eval",
        )[0]
        print('val epoch {} , loss {}, '.format(epoch, loss.item()))
        val_loss.append(loss.item())
        writer.add_scalar('val_loss', loss.item(), epoch)
    writer.close()

    # save model
    checkpoint = {'num_blocks': model.num_layers,
                  'd_model': model.d_model,
                  'd_ff': model.d_ff,
                  'transformer_type': args.transformer_type,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_model_path)


def greedy_decode(model, src_text):
    model.eval()

    src_token = char_vocab.tokenize(src_text, max_len=max_len)
    src_mask = create_encoder_mask(src_token, char_vocab.pad_idx, shape_option=shape_option)
    mem = model.encode(src_token, src_mask)
    ys = torch.full((1, 1), char_vocab.start_idx, dtype=src_token.dtype)
    
    for i in range(max_len):
        # tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        tgt_mask = create_decoder_mask(ys, char_vocab.pad_idx, shape_option=shape_option)
        out = model.decode(mem, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word_idx = torch.max(prob, dim=1)
        next_word_idx = next_word_idx.item()
        ys = torch.cat([ys,torch.tensor([[next_word_idx]], dtype=src_token.dtype)], dim=1)
        if next_word_idx == char_vocab.end_idx:
            break
        # print('tgt_mask:', tgt_mask.shape, tgt_mask)
        # print('out:', out.shape, out)
    # print('ys:', ys)
    out_text = char_vocab.detokenize(ys[0].numpy())
    return out_text


def test_acc(model, device):
    model = model.to(device)
    model.eval()

    # test T times
    T = 100
    cnt = 0
    for i in tqdm(range(T)):
        src_text = char_vocab.random_text(max_len=max_len)
        out_text = greedy_decode(model, src_text)
        # print(f'src:{src_text}, out:{out_text}')
        equal = (src_text == out_text)
        if equal:
            cnt += 1
    acc = cnt / T
    print('total {}, acc {}'.format(T, acc))


# def batch_greedy_decode(model, src, src_mask, max_len, start_symbol):
#     batch_size = src.shape[0]
#     mem = model.encode(src, src_mask)

#     ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)
#     for i in range(max_len - 1):
#         ys_mask = (ys != -1).unsqueeze(-2)
#         sub_mask = subsequent_mask(ys.size(1)).type_as(src.data)
#         ys_mask = ys_mask & sub_mask

#         out = model.decode(mem, src_mask, ys, ys_mask)
#         last_out = out[:, -1]  # (N, D)
#         prob = model.generator(last_out)  # (N, T)
#         _, next_word = torch.max(prob, dim=1)  # (N)
#         # next_word = next_word.data[0]
#         next_word = next_word.unsqueeze(-1)  # (N, 1)
#         ys = torch.cat(
#             # [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
#             [ys, next_word], dim=1
#         )
#     return ys


# def batch_inference(model, inputs_arr):
#     model.eval()

#     # test single
#     src = torch.tensor(inputs_arr)
#     N = src.shape[0]
#     max_len = src.shape[1]
#     src_mask = torch.ones(N, 1, max_len)

#     ys = batch_greedy_decode(model, src, src_mask, max_len, start_symbol=1)
#     return ys


def load_model(model_path):
    checkpoint = torch.load(model_path)
    num_blocks = checkpoint['num_blocks']
    d_model = checkpoint['d_model']
    d_ff = checkpoint['d_ff']
    type = checkpoint['transformer_type']
    state_dict = checkpoint['state_dict']

    if type == 'self':
        model = SeqModelSelf(len(char_vocab), len(char_vocab), N=num_blocks, d_model=d_model, d_ff=d_ff)
    else:
         model = SeqModelAnno(len(char_vocab), len(char_vocab), N=args.num_blocks, d_model=args.d_model, d_ff=args.d_ff)
    model.load_state_dict(state_dict)
    print(f'load model, num_blocks:{num_blocks}, d_model:{d_model}, d_ff:{d_ff}, type:{type}')
    return model


def input_inference(model):
    try:
        while True:
            user_input = input("input str:")
            out_text = greedy_decode(model, user_input)
            print(f'out text:{out_text}')
    except KeyboardInterrupt:
        print('\n exit...')


def test_epoch(model, device,
               batch_size=80, nbatches=20):
    criterion = LabelSmoothing(size=len(char_vocab), padding_idx=0, smoothing=0.0)
    model = model.to(device=device)
    num_total_params = count_total_parameters(model)
    print('model num total params:', num_total_params)

    model.eval()
    loss, _ = run_epoch(
        data_gen_extended_vocab(char_vocab, batch_size, nbatches, max_seq_len=max_len),
        model,
        SimpleLossCompute(model.generator, criterion),
        DummyOptimizer(),
        DummyScheduler(),
        device=device,
        mode="eval",
    )
    print(f'loss: {loss.item()}')


## main flow
if args.mode == 'train':
    train(model, device,
          batch_size=args.batch_size,
          nbatches=args.nbatches,
          epochs=args.epochs,
          save_model_path=args.model_save_path)
    model.to('cpu')
    test_acc(model, device='cpu')
elif args.mode == 'test':
    model = load_model(args.model_save_path)
    print('test acc')
    test_acc(model, device='cpu')

    # print('test epoch')
    # test_epoch(model, device, batch_size=4, nbatches=2)
    print('example inference')
    test_string0 = 'abcdefg'
    test_string1 = 'Hello, World'
    print(f'echo in: {test_string1}')
    model.to('cpu')
    out_text = greedy_decode(model, test_string1)
    print(f'echo out: {out_text}')
elif args.mode == 'input':
    model = load_model(args.model_save_path)
    input_inference(model)
