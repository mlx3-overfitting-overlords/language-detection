
import torch
import tqdm
import pandas
import random
import wandb

import language_detection.dataset as dataset
import model

# Constants
TOTAL_EPOCHS = 2
BATCH_SIZE = 64
MARGIN = 0.2  # Margin for the MarginRankingLoss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 50
VOCAB_SIZE = 8000
HIDDEN_DIM = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="learn-to-search",
    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "embedding_dim": EMBEDDING_DIM,
        "architecture": "TwoTower",
        "dataset": "MS_MARCO_TRAIN",
        "epochs": TOTAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "margin": MARGIN,
        "hidden_dim": HIDDEN_DIM,
    }
)

df = pandas.read_parquet(f'./data.parquet')

# Load sentences from corpus.txt
with open('corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# Remove any newline characters from each sentence
corpus = [sentence.strip() for sentence in corpus]

def custom_collate_fn(batch):
    query_tensor, passage_tensor, labels = zip(*batch)
    
    # Pad queries and passages separately
    padded_queries = torch.nn.utils.rnn.pad_sequence(query_tensor, batch_first=True)
    padded_passages = torch.nn.utils.rnn.pad_sequence(passage_tensor, batch_first=True)
    labels = torch.stack(labels, dim=0)

    return padded_queries, padded_passages, labels

data = df.sample(n=10000, random_state=42).to_dict(orient='records')
ds = dataset.TwoTowerDataset(data)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# Initialize the CBOW model and load its weights
rnn = model.RNNModel(
    embedding_matrix=torch.randn(VOCAB_SIZE, EMBEDDING_DIM),
    hidden_dim=HIDDEN_DIM
)

# Step 2: Load the pre-trained weights
rnn.load_state_dict(torch.load('./weights/v1/rnn_epoch_10.pt'))


# Model, Loss, Optimizer
model = model.TwoTowerModel(rnn.embedding.weight.data).to(DEVICE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

wandb.watch(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# loss_function = torch.nn.MarginRankingLoss(MARGIN)
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCEWithLogitsLoss()

torch.save(model.state_dict(), f"./weights/v1/tt_epoch_0.pt")


# Training loop
model.train()
for epoch in range(TOTAL_EPOCHS):
    total_loss = 0.0
    
    for batch in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", unit="batch"):
        # Reset gradients
        optimizer.zero_grad()
         
        # batch is a portion of the dataset, typically a tuple of inputs and targets
        query_tensor, passage_tensor, label_tensor = batch
        label = label_tensor.float()  # BCEWithLogitsLoss expects float labels
        
        # Forward pass: compute similarity score between query and passage
        scores = model(query_tensor, passage_tensor).squeeze()
        
        # Compute the loss
        loss = loss_function(scores, label)
                
        # Compute the loss between the predictions and the true targets
        # loss = loss_function(query_representation, passage_representation, label)
        
        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        
        # Update model parameters
        optimizer.step()
    
        # Accumulate loss
        total_loss += loss.item()

        wandb.log({"acc": total_loss, "loss": loss})

    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}, Loss: {total_loss/len(dl)}")
    torch.save(model.state_dict(), f"./weights/v1/tt_epoch_{epoch+1}.pt")

wandb.finish()
