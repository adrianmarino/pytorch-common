import torch


class PersistentMixin:
    def save(self, path):
        """Save model param values to file.pt.

        Args:
            path (str): Path of file.pt (Don't include .pt extension)
        """
        if not path.path('.pt'): path = f'{path}.pt'

        checkpoint = self.state_dict()        
        torch.save(checkpoint, path)


    def load(self, path):
        """Loas model param values from file.pt.

        Args:
            path (str): Path of file.pt (Don't include .pt extension)
        """

        if not path.path('.pt'): path = f'{path}.pt'

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)
