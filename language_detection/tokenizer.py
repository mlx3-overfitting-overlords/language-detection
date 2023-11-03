import string
import pandas

def generate_char_ngrams(text, n=3):
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return ngrams

class Tokenizer:
  def __init__(self, corpus=None, freq_threshold=1):
    self.corpus = corpus
    self.freq_threshold = freq_threshold
    self.freq_dist = self.build_freq_dist() if corpus else {}
    self.vocab = self.build_vocab() if corpus else {}
    self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

  def generate_char_ngrams(text, n=3):
      return [text[i:i + n] for i in range(len(text) - n + 1)]

  def build_freq_dist(self):
      freq_dist = {}
      for sentence in self.corpus:
          for word in sentence.split():
              word = self.clean_word(word)
              # Create ngrams for each word and update their frequencies
              for ngram in generate_char_ngrams(word):
                  freq_dist[ngram] = freq_dist.get(ngram, 0) + 1
      return freq_dist
  
  def build_vocab(self):
      tokens = [word.lower() for sentence in self.corpus for word in sentence.split()]
      tokens = [self.clean_word(word) for word in tokens if word]  # Clean and filter out empty words
      
      ngrams = []
      for token in tokens:
          ngrams.extend(generate_char_ngrams(token))  # Generating 3-character ngrams

      # Only keep ngrams that appear more than the threshold times
      vocab = list({ng for ng in ngrams if self.freq_dist.get(ng, 0) > self.freq_threshold})
      return vocab
  
  def save_vocab(self, path):
    with open(path, 'w') as f:
      for word in self.vocab: f.write(word + '\n')
    return self

  def load_vocab(self, path):
    with open(path, 'r') as f: self.vocab = [line.strip() for line in f.readlines()]
    self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
    return self

  def encode(self, sentence):
      cleaned_sentence = self.clean_word(sentence)  # We'll clean the whole sentence
      ngrams = generate_char_ngrams(cleaned_sentence)
      return [self.word2idx[ngram] for ngram in ngrams if ngram in self.word2idx]

  def decode(self, indices):
      return ''.join(self.idx2word[idx] for idx in indices if idx in self.idx2word)


  @staticmethod
  def clean_word(word):
    word = word.lower()
    word = ''.join(char for char in word if char not in string.punctuation)
    word = ''.join(char for char in word if not char.isdigit())
    return word.strip()


if __name__ == '__main__':
  def get_sentences(lang, column=None):
      if column is None:  # set default value for column
          column = lang
      
      dataset_path = f'./datasets/CL_{lang}-en.parquet'
      df = pandas.read_parquet(dataset_path).sample(n=10000,random_state=42)
      
      # Preprocess the text
      sentences = df[column].tolist()

      return sentences


  sentences_fr = get_sentences('fr')
  sentences_en = get_sentences('fr','en')
  sentences_es = get_sentences('es')
  sentences_de = get_sentences('de')
  sentences_it = get_sentences('it')

  # print(sentences_en[:10])


  data = pandas.read_parquet('./datasets/Flores7Lang.parquet')

  long_format = data.melt(value_vars=data.columns)
  corpus =  sentences_en +  sentences_fr +  sentences_es +  sentences_de +  sentences_it + long_format['value'].tolist()

  tknz = Tokenizer(corpus)
  tknz.save_vocab('./vocab.txt')
  tknz.load_vocab('./vocab.txt')
  print(len(tknz.vocab))
  print(tknz.vocab[90:100])