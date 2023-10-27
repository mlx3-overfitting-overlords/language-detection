import torch
import dataset
import pandas
import model
import tqdm

 
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

# print(corpus[:10])
ds = dataset.W2VData(corpus, 3)
dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)


cbow = model.CBOW(len(ds.tokenizer.vocab), 50)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(cbow.parameters(), lr=0.001)

total_epochs = 1

for epoch in range(total_epochs):
  total_loss = 0
  for context, target in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch"):
    optimizer.zero_grad()
    log_probs = cbow(context)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss}")
  torch.save(cbow.state_dict(), f"./weights_old/cbow_epoch_{epoch+1}.pt")