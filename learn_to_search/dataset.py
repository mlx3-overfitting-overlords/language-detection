import torch
import tokenizer
import tqdm

class W2VData(torch.utils.data.Dataset):
    def __init__(self, corpus, window_size=2):
        self.tokenizer = tokenizer.Tokenizer()
        self.data = []
        self.create_data(corpus, window_size)

    def create_data(self, corpus, window_size):
        for sentence in tqdm.tqdm(corpus, desc=f"Creating W2V data", unit="sentence"):
            tokens = self.tokenizer.encode(sentence)
            for i, target in enumerate(tokens):
                context = tokens[max(0, i - window_size):i] + tokens[i + 1:i + window_size + 1]
                if len(context) != 2 * window_size: continue
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

class RNNDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, pad_idx, max_length=None, eos_token='<eos>'):
        self.tokenizer = tokenizer.Tokenizer()
        self.eos_token = eos_token
        self.data = [self.tokenizer.encode(sentence + self.eos_token) for sentence in corpus]
        
        if max_length:
            self.data = [sentence[:max_length] for sentence in self.data]

        self.pad_idx = pad_idx

    def pad_sequence(self, sequence, max_length):
        return sequence + [self.pad_idx for _ in range(max_length - len(sequence))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        input_seq = sentence[:-1]
        target_seq = sentence[1:]
        max_length_in_batch = max([len(seq) for seq in self.data])
        return torch.tensor(self.pad_sequence(input_seq, max_length_in_batch-1)), torch.tensor(self.pad_sequence(target_seq, max_length_in_batch-1))


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = tokenizer.Tokenizer()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        query = row['query_text']
        passage = row['passage_text']
        label = row['label']

        query_tensor = torch.tensor(self.tokenizer.encode(query))
        passage_tensor = torch.tensor(self.tokenizer.encode(passage))
        label_tensor = torch.tensor(label, dtype=torch.float32)  # Ensure this is the correct type

        return query_tensor, passage_tensor, label_tensor
