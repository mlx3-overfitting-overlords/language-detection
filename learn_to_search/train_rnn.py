import torch
import language_detection.dataset as dataset
import model
import tqdm
import random
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

EMBEDDING_DIM = 50
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_DIM = 100

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="learn-to-search",
    
    # track hyperparameters and run metadata
    config={
      "learning_rate": LEARNING_RATE,
      "embedding_dim": EMBEDDING_DIM,
      "architecture": "RNN",
      "dataset": "MS_MARCO_TRAIN",
      "epochs": TOTAL_EPOCHS,
      "batch_size": BATCH_SIZE,
    }
)

def collate_fn(batch):
    # Just pads the sequences in the batch
    return pad_sequence(batch, batch_first=True)

# Load sentences from corpus.txt
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# Remove any newline characters from each sentence
corpus = [sentence.strip() for sentence in corpus]
corpus = random.sample(corpus, 10000)

# # SentencePiece tokenizer does use a padding token. By default, its token is <pad>, and the ID for this token is 0.
# pad_idx = 0
# ds = dataset.RNNDataset(corpus, pad_idx)  # Assuming you've defined RNNDataset as shown earlier
def collate_fn(batch):
    # Separate sequences and targets
    sequences, targets = zip(*batch)

    # Pad sequences and targets
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)

    return padded_sequences, padded_targets

ds = dataset.RNNDataset(corpus, pad_idx=0)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

vocab_size = len(ds.tokenizer.vocab)
# Initialize the CBOW model and load its weights
cbow = model.CBOW(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)
cbow.load_state_dict(torch.load('./weights/v1/cbow_epoch_1.pt'))

# Initialize the RNN model using the embeddings from CBOW
rnn = model.RNNModel(cbow.embeddings.weight.data, hidden_dim=HIDDEN_DIM)
wandb.watch(rnn)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr = LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
torch.save(rnn.state_dict(), f"./weights/v1/rnn_epoch_0.pt")

for epoch in range(TOTAL_EPOCHS):
    total_loss = 0
    num_batches = 0

    for sequence, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", unit="batch"):
        optimizer.zero_grad()
        output = rnn(sequence)
        # Flatten outputs and targets for the loss function
        loss = loss_function(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        wandb.log({"loss": loss.item()})

    
    # After all batches, log the average loss for the epoch
    average_loss = total_loss / num_batches
    wandb.log({"epoch": epoch, "average_loss": average_loss})
    
    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}, Average loss: {average_loss}")
    torch.save(rnn.state_dict(), f"./weights/v1/rnn_epoch_{epoch+1}.pt")

wandb.finish()