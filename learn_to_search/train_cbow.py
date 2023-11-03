import torch
import language_detection.dataset as dataset
import model
import tqdm
import wandb
import random

EMBEDDING_DIM = 50
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 10
BATCH_SIZE = 128
W2V_WINDOW_SIZE = 3

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="learn-to-search",
    
    # track hyperparameters and run metadata
    config={
      "learning_rate": LEARNING_RATE,
      "embedding_dim": EMBEDDING_DIM,
      "architecture": "CBOW",
      "dataset": "MS_MARCO_TRAIN",
      "epochs": TOTAL_EPOCHS,
      "batch_size": BATCH_SIZE,
      "w2v_window_size": W2V_WINDOW_SIZE
    }
)

# Load sentences from corpus.txt
with open('corpus.txt', 'r', encoding='utf-8') as f:
  corpus = f.readlines()

# Remove any newline characters from each sentence
corpus = [sentence.strip() for sentence in corpus]
corpus = random.sample(corpus, 10000)

ds = dataset.W2VData(corpus, W2V_WINDOW_SIZE)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

cbow = model.CBOW(len(ds.tokenizer.vocab), EMBEDDING_DIM)
wandb.watch(cbow)

loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

for epoch in range(TOTAL_EPOCHS):
    total_loss = 0
    # Use a variable to track the number of batches
    num_batches = 0
    for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", unit="batch"):
        optimizer.zero_grad()
        log_probs = cbow(context)
        loss = loss_function(log_probs, target)
        loss.backward()
        
        # Potential improvement: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(cbow.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        # Log the average loss for the current batch
        wandb.log({"loss": loss.item()})
    
    # After all batches, log the average loss for the epoch
    average_loss = total_loss / num_batches
    wandb.log({"epoch": epoch, "average_loss": average_loss})
    
    # Step the scheduler
    scheduler.step()

    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}, Average loss: {average_loss}")
    torch.save(cbow.state_dict(), f"./weights/v2/cbow_epoch_{epoch+1}.pt")

wandb.finish()
