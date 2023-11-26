
from transformer_annotated import *
from transformer_self import TransformerModel


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x: (N, T, D)
        # out:(N, T, V)
        out = self.proj(x)
        return nn.functional.log_softmax(out, dim=-1)


def SeqModelAnno(num_src_vocab, num_tgt_vocab,  N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # num_src_vocab 是src的词表数量, tgt_vocab是tgt的词表数量
    # d_model是词嵌入式向量的大小，h是多头注意力机制的数量， N是transformer block num
    # d_ff是前向反馈层的中间层数
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, num_src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, num_tgt_vocab), c(position)),
        Generator(d_model, num_tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class SeqModelSelf(nn.Module):
    def __init__(self, num_src_vocab, num_tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_seq_length=5000):
        super(SeqModelSelf, self).__init__()
        self.transformer = TransformerModel(src_vocab_size=num_src_vocab, tgt_vocab_size=num_tgt_vocab, 
                                            embed_dim=d_model, nhead=h, num_encoder_layers=N, num_decoder_layers=N, 
                                            dim_feedforward=d_ff, max_seq_length=max_seq_length, dropout=dropout)
        self.generator = Generator(d_model, num_tgt_vocab)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = N
        self._reset_parameters()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.transformer(src, tgt, src_mask, tgt_mask)  # generator will be used in simple loss
        # return self.generator(self.transformer(src, tgt, src_mask, tgt_mask))
    
    def encode(self, src, src_mask=None):
        return self.transformer.encode(src, src_mask)

    def decode(self, memory, tgt, tgt_mask=None):
        return self.transformer.decode(memory, tgt, tgt_mask)

    def _reset_parameters(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


if __name__ == '__main__':
    from char_gen_dataloader import create_decoder_mask
    def inference_test():
        # test_model = seq_model(num_src_vocab=10, num_tgt_vocab=10, N=1)
        test_model = SeqModelAnno(num_src_vocab=10, num_tgt_vocab=10, N=1)
        test_model.eval()
        src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
        # src mask will broadcast
        # src_mask = torch.ones(1, 1, 10)
        src_mask = torch.ones(1, 10, 10)

        print('model encode')
        memory = test_model.encode(src, src_mask)
        print('model encode finished...')
        ys = torch.zeros(1, 1).type_as(src)

        for i in range(9):
            ys_mask = create_decoder_mask(tgt=ys, pad_idx=0)
            # print('ys shape:', ys.shape)
            # print(f'ys mask shape: {ys_mask.shape},  \n {ys_mask}')
            # print('ys:\n', ys)
            print('ys_mask:\n', ys_mask)
            out = test_model.decode(
                memory, mem_mask=None, tgt=ys, tgt_mask=ys_mask
            )
            prob = test_model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )

        print(f"iter {i}  Untrained Model Prediction:", ys)
    
    
    seed = 42
    torch.manual_seed(seed)
    inference_test()
    # for _ in range(10):
    # inference_test()
