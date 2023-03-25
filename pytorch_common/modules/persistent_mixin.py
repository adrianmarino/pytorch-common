import torch


class PersistentMixin:
    def save(self, path):
        checkpoint = self.state_dict()
        torch.save(checkpoint, f'{path}.pt')


    def load(self, path):
        checkpoint = torch.load(f'{path}.pt')
        self.load_state_dict(checkpoint)
