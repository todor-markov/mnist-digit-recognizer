import os
import click
import numpy as np

from util import get_mnist

import torch
import torch.nn as nn
import torch.optim as optim


class View(nn.Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(*self.shape)


@click.command()
@click.option('--data-dir', type=str, default='data/mnist')
@click.option('--n-episodes', '-e', type=int, default=5)
@click.option('--batch-size', '-b', type=int, default=128)
@click.option('--log-every', type=int, default=1)
def main(data_dir, n_episodes, batch_size, log_every):
    trX, teX, trY, teY = get_mnist(os.path.expanduser(f"~/{data_dir}"))
    trX = torch.Tensor(trX)
    teX = torch.Tensor(teX)
    trY = torch.Tensor(trY).to(torch.long)
    teY = torch.Tensor(teY).to(torch.long)
    n_train_samples = trX.shape[0]
    
    image_classifier = nn.Sequential(
        View(-1, 1, 28, 28),
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        View(-1, 7 * 7 * 32),
        nn.Linear(7 * 7 * 32, 48),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(48, 10))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(image_classifier.parameters())

    for ep in range(n_episodes):
        sortidxs = np.random.permutation(n_train_samples)
        trX = trX[sortidxs]
        trY = trY[sortidxs]

        for batch_start in range(0, n_train_samples, batch_size):
            batch_end = batch_start + batch_size
            optimizer.zero_grad()
            outputs = image_classifier(trX[batch_start:batch_end])
            loss = criterion(outputs, trY[batch_start:batch_end])
            loss.backward()
            optimizer.step()

        print(f'Episode {ep + 1}')
        if (ep + 1) % log_every == 0:
            eval_outputs = image_classifier(teX)
            eval_loss = criterion(eval_outputs, teY)
            eval_acc = (eval_outputs.argmax(dim=1) == teY).to(torch.float32).mean()
            print(f'Loss: {eval_loss}. Accuracy: {eval_acc}')


if __name__ == '__main__':
	main()
