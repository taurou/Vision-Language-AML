import torch
from experiments.utils import *
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt): #TODO 
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])

        # Optimizers for all parts of the network.
        self.criterion_CEL = torch.nn.CrossEntropyLoss()
        self.criterion_EL = EntropyLoss()
        self.criterion_L2L = L2Loss() 

        #TODO Weights 
        self.w1 = opt["weights"][0]
        self.w2 = opt["weights"][1]
        self.w3 = opt["weights"][2]
        self.alpha = opt["weights"][3]
        #self.future-paramter = opt["weights"][4] riservato per il futuro clip o chicchessia
        print("Domain Disentangle parameters: \n","w1: ", self.w1, "w2: ", self.w2, "w3: ", self.w3, "alpha: ", self.alpha)

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

    def train_iteration(self, data, targetDomain = False):

        self.zero_gradients()
        
        if(targetDomain == False):
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)           
            domain_labels = torch.zeros(len(x), dtype=torch.long).to(self.device) 
            category_pred = self.model(x, step = 0)

        else:
            x, _ = data
            x = x.to(self.device)
            domain_labels = torch.ones(len(x), dtype=torch.long).to(self.device) 
            category_loss = 0

        confuse_domain = self.model(x, step = 1) #TODO check if softmax is necessary somewhere
        confuse_domain_loss = -self.w1*self.criterion_EL(confuse_domain)*self.alpha

        domain_pred = self.model(x, step = 2)
        domain_loss = self.w2*self.criterion_CEL(domain_pred, domain_labels)
        
        confuse_category = self.model(x, step = 3)
        confuse_category_loss = -self.w2*self.criterion_EL(confuse_category)*self.alpha

        Fg, Rfg = self.model(x, step = 4) #return reconstructor features (Rfg) and extracted features Fg
        reconstructor_loss = self.w3*self.criterion_L2L(Rfg, Fg)

        loss = category_loss + confuse_domain_loss + domain_loss + confuse_category_loss + reconstructor_loss
        return loss.item()



    def validate(self, loader): #TODO Comment by tauro: during validation phase, we should pass only the source domain? Because, """""theoretically""""" it's the only data with labels and thus can be used to check on the model.
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, step = 0 )
                loss += self.criterion_CEL(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss




'''
        (Fg, Cc, Cd, Ccd, Cdc, Rfg) = logits = self.model(x)
        loss = self.criterion(logits, y) #TODO extract and handle loss

        #loss from the paper:
        #loss = w1 * loss_class + w2 * loss_domain + w3 * loss_reconstructor
        #loss_class = loss_class-CROSSENTROPY + alpha * loss_class-ENTROPY
        #loss_domain = loss_domain-CROSSENTROPY + alpha * loss_domain-ENTROPY
        #loss_reconstruction = 

        loss_category = self.criterion_CEL(Cc, y) + self.alpha * -self.criterion_EL(Ccd)
        loss_domain = self.criterion_CEL(Cc) + self.alpha * -self.criterion_EL(Cdc) #TODO domain label is missing in cross-entropy loss.
        
        loss_reconstructor = self.criterion_L2L(Rfg, Fg)
        loss = self.w1 * loss_category + self.w2 * loss_domain + self.w3 * loss_reconstructor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


'''