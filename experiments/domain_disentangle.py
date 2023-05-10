import torch
from utils import EntropyLoss
class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt): #TODO 
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleExperiment()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # TODO Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion_CEL = torch.nn.CrossEntropyLoss()
        self.criterion_EL = EntropyLoss()
        self.criterion_MSE = torch.nn.MSELoss() #TODO L2 loss or MSE?

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
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        (Fg, Cc, Cd, Ccd, Cdc, Rfg) = logits = self.model(x)
        loss = self.criterion(logits, y) #TODO extract and handle loss

        #loss from the paper:
        #loss = w1 * loss_class + w2 * loss_domain + w3 * loss_reconstructor
        #loss_class = loss_class-CROSSENTROPY + loss_class-ENTROPY
        #loss_domain = loss_domain-CROSSENTROPY + loss_domain-ENTROPY


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss