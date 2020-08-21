import torch
import torchtext
from fastNLP.models import BertForSequenceClassification
from torchtext.experimental.datasets import AG_NEWS
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split
import argparse
import random
import numpy as np

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label - 1

class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit

        batch_size = 32
        embed_dim = 64
        epochs = 5
        num_labels = 4

        # TODO: Can we cache this in setup?
        train_dataset, _ = AG_NEWS(ngrams=1)

        bert_embed = torch.nn.EmbeddingBag(len(train_dataset.vocab), embed_dim)
        self.model = BertForSequenceClassification(bert_embed, num_labels=num_labels).to(self.device)

        if self.jit:
            self.model = torch.jit.script(self.model)

        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        data = DataLoader(train_dataset, batch_size=batch_size, collate_fn=generate_batch)
        self.text, self.offsets, self.cls = [x.to(self.device) for x in next(iter(data))]

    def get_module(self):
        return self.model, (self.text, self.offsets)

    def eval(self, niter=1):
        with torch.no_grad():
            for _ in range(niter):
                output = self.model(self.text, self.offsets)
                loss = self.criterion(output, self.cls)

    def train(self, niter=1):
        for _ in range(niter):
            self.optimizer.zero_grad()
            output = self.model(self.text, self.offsets)
            loss = self.criterion(output, self.cls)
            loss.backward()
            self.optimizer.step()

        # Adjust the learning rate
        # Should we benchmark this?  It's run once per epoch
        # self.scheduler.step()

if __name__ == "__main__":
    m = Model(device="cuda:0", jit=False)
    model, example_inputs = m.get_module()
    model(*example_inputs)
    m.train()
    m.eval()
