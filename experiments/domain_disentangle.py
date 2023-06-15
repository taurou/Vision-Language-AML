import torch
from experiments.utils import *
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt): #TODO 
        # Utils
        self.opt = opt  # We store options dict (hyperparameters)
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel(opt)    # See point 2. of the project
                                                    # We create an istance of the class DomainDisentangleModel
        self.model.train()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])

        # Optimizers for all parts of the network.
        self.criterion_CEL = torch.nn.CrossEntropyLoss()    
        # Differently from the baseline, we have 3 different loss functions, in particular these two are added:
        self.criterion_EL = EntropyLoss()
        self.criterion_L2L = L2Loss() 

        # Weights: w1, w2, w3, alpha. They are used to weight the three loss functions.
        self.w1 = opt["weights"][0]
        self.w2 = opt["weights"][1]
        self.w3 = opt["weights"][2]
        self.alpha = opt["weights"][3]
        print("Domain Disentangle parameters: \n","w1: ", self.w1, "w2: ", self.w2, "w3: ", self.w3, "alpha: ", self.alpha)


    def categoryClassifierTraining(self, train = False):
        if self.opt["disable_classifier"]:
            for param in self.model.category_classifier.parameters():
                param.requires_grad = train

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

        self.optimizer.zero_grad()  # Reset gradients accumulation, otherwise they would be summed to the previous ones.

        if not self.opt["dom_gen"]: 
        
            if(targetDomain == False):
                x, y = data # x is the image tensor, y is the category label tensor.
                x = x.to(self.device)
                y = y.to(self.device)           
                domain_labels = torch.zeros(len(x), dtype=torch.long).to(self.device) # Domain labels is a tensor of zeros, because we are in the source domain.  
                self.categoryClassifierTraining(train = True) #Enable the category classifier training since no loss will be computed
            else:
                x, _ = data # x is the image tensor, y is the category label tensor. The _ is here because we MUST NOT use the category label for the unsupervised learning of the target domain features
                x = x.to(self.device)
                domain_labels = torch.ones(len(x), dtype=torch.long).to(self.device)  # Domain labels is a tensor of ones, because we are in the target domain.
                self.categoryClassifierTraining(train = False) #Disable the category classifier training since no loss will be computed

            (Fg, Cc, Cd, Ccd, Cdc, Rfg, _) = self.model(x) # self.model(x) returns a tuple of 7 elements, which are the outputs of the 7 modules of the network of the class DomainDisentangleModel.
            # The _ because self.model also returns the features extracted by the domain encoder, not necessary here.

            category_loss = 0 if targetDomain == True else self.criterion_CEL(Cc, y) #TODO rivedere ordine dei parametri
            
            confuse_domain_loss = -self.criterion_EL(Ccd)           

            domain_loss = self.criterion_CEL(Cd, domain_labels)     # We use the cross entropy loss function to minimize the error of the domain classifier, so that it can distinguish between the two domains.

            confuse_category_loss = -self.criterion_EL(Cdc)

            reconstructor_loss = self.criterion_L2L(Rfg, Fg)

            loss = self.w1*(category_loss + self.alpha*confuse_domain_loss) + self.w2*(domain_loss + self.alpha*confuse_category_loss) + self.w3*reconstructor_loss
            loss.backward()
            self.optimizer.step()
            return loss.item()

        else: #Domain Generalization

            x, y, domain_labels = data #x is the image tensor, y is the category label tensor.
            x = x.to(self.device)
            y = y.to(self.device)  
            domain_labels = domain_labels.to(self.device)         

            
            (Fg, Cc, Cd, Ccd, Cdc, Rfg, _) = self.model(x)  #the _ because self.model also returns the features extracted by the domain encoder, not necessary here.

            category_loss = self.criterion_CEL(Cc, y) #TODO rivedere ordine dei parametri
            
            confuse_domain_loss = -self.criterion_EL(Ccd)

            domain_loss = self.criterion_CEL(Cd, domain_labels)

            confuse_category_loss = -self.criterion_EL(Cdc)

            reconstructor_loss = self.criterion_L2L(Rfg, Fg)

            loss = self.w1*(category_loss + self.alpha*confuse_domain_loss) + self.w2*(domain_loss + self.alpha*confuse_category_loss) + self.w3*reconstructor_loss
            loss.backward()
            self.optimizer.step()

            return loss.item()

    def validate(self, loader): #TODO Comment by tauro: during validation phase, we should pass only the source domain? Because, """""theoretically""""" it's the only data with labels and thus can be used to check on the model.
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for data in loader: # _ because here we don't need the domain labels. 
                x = data[0]
                y = data[1]
                x = x.to(self.device)
                y = y.to(self.device)

                (_, Cc,_, _, _, _, _) = self.model(x) #The series of _ here is used because the only parameter we need are the predicted categories.
                loss += self.criterion_CEL(Cc, y)
                pred = torch.argmax(Cc, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss

    