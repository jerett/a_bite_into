import torch
import random
import string


# def subsequent_mask(size):
#     "Mask out subsequent positions."
#     attn_shape = (1, size, size)
#     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
#         torch.uint8
#     )
#     return subsequent_mask == 0

def create_encoder_mask(src, pad_idx, shape_option=1):
    """
    Creates a mask for padding tokens in the encoder.

    Args:
    - src: tensor of shape (N, T)
    - pad_idx: index of the padding token
    - shape_option: int, if 1 returns shape (N, 1, 1, T), if 2 returns shape (N, 1, T)

    Returns:
    - a mask of shape based on shape_option
    """
    src_mask = (src != pad_idx).unsqueeze(1)  # Shape: (batch_size, 1, src_seq_length)
    if shape_option == 1:
        src_mask = src_mask.unsqueeze(2)  # Shape: (batch_size, 1, 1, src_seq_length)
    return src_mask

# def create_encoder_mask(src, pad_idx):
#     """
#     Creates a mask for padding tokens in the encoder.

#     Args:
#     - src: tensor of shape (N, T)
#     - pad_idx: index of the padding token

#     Returns:
#     - a mask of shape (N, 1, 1, T)
#     """
#     src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, src_seq_length)
#     return src_mask


# def create_encoder_mask2(src, pad_idx):
#     """
#     Creates a mask for padding tokens in the encoder.

#     Args:
#     - src: tensor of shape (N, T)
#     - pad_idx: index of the padding token

#     Returns:
#     - a mask of shape (N, 1, T)
#     """
#     src_mask = (src != pad_idx).unsqueeze(-2)  # Shape: (batch_size, 1, src_seq_length)
#     return src_mask

def create_decoder_mask(tgt, pad_idx, shape_option=1):
    """
    Creates a combined mask for the padding tokens and future tokens in the decoder.

    Args:
    - tgt: tensor of shape (N, T)
    - pad_idx: index of the padding token
    - shape_option: int, if 1 returns shape (N, 1, T, T), if 2 returns shape (N, T, T)

    Returns:
    - a mask of shape based on shape_option
    """
    
    N, T = tgt.shape

    # Padding mask - False where tgt is padding
    pad_mask = (tgt != pad_idx).unsqueeze(1)  # Shape: (N, 1, T)

    # Look-ahead mask - False below diagonal and True above
    look_ahead_mask = torch.tril(torch.ones((T, T), device=tgt.device), diagonal=0).bool()
    look_ahead_mask = look_ahead_mask.unsqueeze(0)  # Shape: (1, T, T)

    # Combined mask - True where either mask is True
    if shape_option == 1:
        pad_mask = pad_mask.unsqueeze(2) # Shape: (N, 1, 1, T)
    combined_mask = pad_mask & look_ahead_mask  # Shape: (N, 1, T, T)
    return combined_mask
    
    # T = tgt.size(1)
    # pad_mask = (tgt == pad_idx).unsqueeze(1)  # Shape: (N, 1, T)
    # look_ahead_mask = torch.triu(torch.ones((T, T), device=tgt.device), diagonal=1).type(torch.bool)

    # if shape_option == 1:
    #     pad_mask = pad_mask.unsqueeze(2)  # Shape: (N, 1, 1, T)
    #     look_ahead_mask = look_ahead_mask.unsqueeze(0)  # Shape: (1, T, T)
    #     combined_mask = pad_mask | look_ahead_mask
    # else:
    #     combined_mask = pad_mask | look_ahead_mask.unsqueeze(0)  # Shape: (N, T, T)
    # return combined_mask
    # pad_mask = (tgt != pad_idx).unsqueeze(-2)  # (N, 1, T)
    # T = tgt.size(-1)
    # look_ahead_mask = (torch.triu(torch.ones((T, T)), diagonal=1).type(torch.uint8) == 0)
    # return pad_mask & look_ahead_mask


# def create_decoder_mask(tgt, pad_idx):
#     """
#     Creates a combined mask for the padding tokens and future tokens in the decoder.

#     Args:
#     - tgt: tensor of shape (N, T)
#     - pad_idx: index of the padding token

#     Returns:
#     - a mask of shape (N, 1, T, T)
#     """
#     tgt_len = tgt.size(1)

#     # Padding mask for the targets
#     pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, tgt_seq_length)

#     # Look-ahead mask for future tokens
#     look_ahead_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).type(torch.bool)
#     look_ahead_mask = look_ahead_mask.unsqueeze(0)  # Shape: (1, tgt_seq_length, tgt_seq_length)

#     combined_mask = pad_mask | look_ahead_mask  # Combine the two masks
#     return combined_mask


# def create_decoder_mask2(tgt, pad_idx):
#     """
#     Creates a combined mask for the padding tokens and future tokens in the decoder.

#     Args:
#     - tgt: tensor of shape (N, T)
#     - pad_idx: index of the padding token

#     Returns:
#     - a mask of shape (N, T, T)
#     """
#     # tgt_len = tgt.size(1)
#     # look_ahead_mask = torch.triu(torch.ones((tgt_len, tgt_len)), diagonal=1).type(torch.bool)
#     # padding_mask = (tgt == pad_token_idx).type(torch.bool)
#     # return look_ahead_mask | padding_mask

#     pad_mask = (tgt != pad_idx).unsqueeze(-2)  # (N, 1, T)
#     T = tgt.size(-1)
#     look_ahead_mask = (torch.triu(torch.ones((T, T)), diagonal=1).type(torch.uint8) == 0)
#     return pad_mask & look_ahead_mask


# 创建扩展的词汇表
class ExtendedVocab:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        for idx, char in enumerate(string.ascii_letters + string.digits + string.punctuation + " ", start=3):
            self.vocab[char] = idx
        self.reverse_vocab = {idx: char for char, idx in self.vocab.items()}
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
    
    def tokenize(self, text, max_len):
        chars = ["<sos>"] + list(text) + ["<eos>"]  # Add <sos> and <eos>
        data = torch.full((1, max_len),  self.pad_idx, dtype=torch.long)
        idx = [self.vocab[char] for char in chars]  # Map to indices
        data[0, :len(idx)] = torch.tensor(idx, dtype=torch.long)
        return data

    def detokenize(self, indices):
        inner_indices = indices[1:-1]  # remove start and stop indices
        chars = [self.reverse_vocab[idx] for idx in inner_indices]
        de_text = ''.join(chars)
        return de_text
    
    def random_text(self, max_len):
        all_chars = list(self.vocab.keys())[3:]  # Exclude special tokens
        # Random sequence length
        seq_len = random.randint(2, max_len - 1)
        # Generate random character sequence
        random_chars = random.choices(all_chars, k=seq_len - 1)
        text = "".join(random_chars)
        return text

    def __getitem__(self, char):
        return self.vocab.get(char, self.vocab["<pad>"])

    def __len__(self):
        return len(self.vocab)

        
# Batch 类
class Batch:
    def __init__(self, src, tgt=None, chars_data=None, pad_idx=0, shape_option=1):  # pad token 默认为 0
        self.src = src
        self.src_mask = create_encoder_mask(src, pad_idx, shape_option=shape_option)
        
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = create_decoder_mask(self.tgt, pad_idx, shape_option=shape_option)
            self.chars_data = chars_data
            self.ntokens = (self.tgt_y != pad_idx).data.sum()


def data_gen_extended_vocab(vocab, batch_size, nbatches, max_seq_len=128, shape_option=1):
    V = len(vocab)  # Size of the vocabulary
    all_chars = list(vocab.vocab.keys())[3:]  # Exclude special tokens
 
    for _ in range(nbatches):
        data = torch.full((batch_size, max_seq_len), vocab.pad_idx, dtype=torch.long)
        chars_data = []

        for i in range(batch_size):
            # Random sequence length
            seq_len = random.randint(2, max_seq_len - 1)
            # Generate random character sequence
            random_chars = random.choices(all_chars, k=seq_len - 1)
            random_chars = ["<sos>"] + random_chars + ["<eos>"]  # Add <sos> and <eos>
            indexed_sequence = [vocab.vocab[char] for char in random_chars]  # Map to indices
            data[i, :len(indexed_sequence)] = torch.tensor(indexed_sequence, dtype=torch.long)
            chars_data.append(random_chars)

        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, chars_data, vocab.pad_idx, shape_option)


if __name__ == '__main__':
    # 使用示例
    extended_vocab = ExtendedVocab()
    for batch in data_gen_extended_vocab(extended_vocab, batch_size=5, nbatches=3, max_seq_len=8, shape_option=0):
        print('data chars:\n', batch.chars_data)
        print("Source Batch:\n", batch.src)
        print("Batch Source Mask Shape:\n", batch.src_mask.shape)
        print("Batch Source Mask:\n", batch.src_mask)
        print("Target Batch:\n", batch.tgt)
        print("Target Mask Shape:\n", batch.tgt_mask.shape)
        print("Batch Target Mask:\n", batch.tgt_mask)
        print("Number of tokens (excluding padding):\n", batch.ntokens)
    idx = extended_vocab.tokenize('abcdefg', max_len=16)
    print(idx)
