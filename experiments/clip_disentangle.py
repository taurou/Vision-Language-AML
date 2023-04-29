import torch

class CLIPDisentangleExperiment: # See point 4. of the project
    
    def __init__(self, opt):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')

    def validate(self, loader):
        raise NotImplementedError('[TODO] Implement CLIPDisentangleExperiment.')