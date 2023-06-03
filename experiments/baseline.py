import torch
from models.base_model import BaselineModel

class BaselineExperiment: # See point 1. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt  # We store options dict (hyperparameters)
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')   

        # Setup model
        self.model = BaselineModel()            # See point 1. of the project
                                                # We create an istance of the class BaselineModel
        self.model.train()                      # We set the model in training mode
        self.model.to(self.device)              # We move the model to the device (cpu or gpu)
        
        for param in self.model.parameters():   
            param.requires_grad = True

        # Setup optimization procedure, the optimizer Adam and loss criterion CrossEntropyLoss are initialized
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

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

    def train_iteration(self, data): # It's the single iteration of the training loop
        x, y, _ = data #_ because it stores the information about the domain. It's not necessary here. 
        x = x.to(self.device) # x is the input image
        y = y.to(self.device) # y is the label

        logits = self.model(x)  # logits is the output of the model, it's a vector of 7 elements
                                # each element is the probability of the input image to belong to a specific class
                                # the index of the element with the highest probability is the predicted class
        
        loss = self.criterion(logits, y)    # We compute the loss between the output of the model and the label

        self.optimizer.zero_grad()          # We set the gradients to zero to avoid accumulating them at each iteration
        loss.backward()                     # We compute the gradients, the loss is backpropagated through the network
                                            # Consente di determinare come ogni parametro del modello contribuisce all'errore complessivo della rete neurale.
        self.optimizer.step()               # We update the parameters of the model using the gradients, 
                                            # the optimizer Adam is used to update the parameters
        
        return loss.item()

    def validate(self, loader):
        self.model.eval()       # We set the model in evaluation mode
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():   # We don't need to compute the gradients during validation because we don't update the parameters
            for data in loader: # We iterate over the validation set, provided by the loader 
                x = data[0]     
                y = data[1]     
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1) # We compute the predicted class as the index of the element with the highest probability

                accuracy += (pred == y).sum().item()    # We compute the accuracy as the number of correct predictions divided by the number of samples
                count += x.size(0)                      # We count the number of samples

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()                              # We set the model in training mode again to prepare for the next iteration
        return mean_accuracy, mean_loss