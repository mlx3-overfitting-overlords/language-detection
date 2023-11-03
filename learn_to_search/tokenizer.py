# tokenizer.py
import sentencepiece as spm

model_prefix="spm"
vocab_size=8000

def train():
    # Convert corpus list of sentences to a single string and save to a file

    # Train the SentencePiece model
    spm.SentencePieceTrainer.Train(
        f'--input=corpus.txt --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=unigram'
    )

class Tokenizer:
    def __init__(self, model_path="spm.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    @property
    def vocab(self):
        return {self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())}


if __name__ == '__main__':
      train()
