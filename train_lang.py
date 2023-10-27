import torch
import dataset
import pandas
import model


data_flores = pandas.read_parquet('./datasets/Flores7Lang.parquet')
data_fr = pandas.read_parquet('./datasets/CL_fr-en.parquet').sample(n=10000,random_state=42)
data_es = pandas.read_parquet('./datasets/CL_es-en.parquet').sample(n=10000,random_state=42)
data_de = pandas.read_parquet('./datasets/CL_de-en.parquet').sample(n=10000,random_state=42)
data_it = pandas.read_parquet('./datasets/CL_it-en.parquet').sample(n=10000,random_state=42)


# Rename the columns
mapping = {
    'fr': 'fra',
    'es': 'spa',
    'de': 'deu',
    'it': 'ita',
    'en': 'eng'  # Assuming you also want to change 'en' to 'eng'
}

data_fr = data_fr.rename(columns=mapping)
data_es = data_es.rename(columns=mapping)
data_de = data_de.rename(columns=mapping)
data_it = data_it.rename(columns=mapping)

# Concatenate
data = pandas.concat([data_flores, data_fr, data_de, data_es, data_it], ignore_index=True)
data = data.dropna(subset=['fra', 'deu', 'spa', 'ita', 'eng'])

# print(data.info())


ds = dataset.LangData(data)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

cbow = model.CBOW(len(ds.tknz.vocab), 50)
cbow.load_state_dict(torch.load('./weights_old/cbow_epoch_1.pt'))
lang = model.Language(cbow.embeddings.weight.data, 7)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lang.parameters(), lr=0.001)
torch.save(lang.state_dict(), f"./weights_old/lang_epoch_0.pt")

total_epochs = 3
for epoch in range(total_epochs):
  for sentence, target, _ in dl:
    optimizer.zero_grad()
    log_probs = lang(sentence)
    loss = loss_function(log_probs, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{total_epochs}, Loss: {loss.item()}")
  torch.save(lang.state_dict(), f"./weights_old/lang_epoch_{epoch+1}.pt")