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

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        # Optimizers for all parts of the network.
        self.optimizer0 = torch.optim.Adam(list(self.model.feature_extractor.parameters()) + 
                                           list(self.model.category_encoder.parameters()) + 
                                           list(self.model.category_classifier.parameters()) , lr=opt['lr'])
        
        #we freeze the domain classifier DC and train the disentangler to generate fcs (category specific features)
        #with the objective to fool the domain classifier
        self.optimizer1 = torch.optim.Adam(self.model.category_encoder.parameters() , lr=opt['lr'])
       
        self.optimizer2 = torch.optim.Adam(list(self.model.feature_extractor.parameters()) + 
                                           list(self.model.domain_encoder.parameters()) + 
                                           list(self.model.domain_classifier.parameters()) , lr=opt['lr'])
        
        #With a well-trained classifier C, we freeze the parameter weights of C and
        #train the disentangler to generate fds to confuse the category classifier C
        self.optimizer3 = torch.optim.Adam(self.model.domain_encoder.parameters(), lr=opt['lr'])
        
        #maybe the feature extractor is not needed to be optimized because in this step i'm optimizing elsewhere.
        #  In fact the features coming from the reconstruct should be as similar as possible as the features coming from the feature extractor 
        self.optimizer4 = torch.optim.Adam( 
                                            list(self.model.domain_encoder.parameters()) +  
                                            list(self.model.category_encoder.parameters()) + 
                                            list(self.model.reconstructor.parameters()) , lr=opt['lr'])
        
        self.criterion_CEL = torch.nn.CrossEntropyLoss()
        self.criterion_EL = EntropyLoss()
        self.criterion_L2L = L2Loss() 

        #TODO Weights 
        self.w1 = 1
        self.w2 = 2
        self.w3 = 3
        self.alpha = 4

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = [self.optimizer0.state_dict(), self.optimizer1.state_dict(), self.optimizer2.state_dict(), self.optimizer3.state_dict(), self.optimizer4.state_dict()]

        torch.save(checkpoint, path)


    def zero_gradients(self):
        self.optimizer0.zero_grad()
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        self.optimizer4.zero_grad()
    
    
    def load_checkpoint(self, path):
        
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer0.load_state_dict(checkpoint['optimizer'][0])
        self.optimizer1.load_state_dict(checkpoint['optimizer'][1])
        self.optimizer2.load_state_dict(checkpoint['optimizer'][2])
        self.optimizer3.load_state_dict(checkpoint['optimizer'][3])
        self.optimizer4.load_state_dict(checkpoint['optimizer'][4])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, targetDomain = False):

        self.zero_gradients()
        
        if(targetDomain == False):
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)           
            domain_labels = torch.zeros(len(x), dtype=torch.long).to(self.device) #TODO must check if this works

            category_pred = self.model(x, step = 0)
            category_loss = self.w1*self.criterion_CEL(category_pred, y)
            category_loss.backward()
            self.optimizer0.step()
            self.zero_gradients()

        else:
            x, _ = data
            x = x.to(self.device)
            domain_labels = torch.ones(len(x), dtype=torch.long).to(self.device) #TODO must check if this works
            category_loss = 0

        confuse_domain = self.model(x, step = 1)
        confuse_domain_loss = -self.w1*self.criterion_EL(confuse_domain)*self.alpha
        confuse_domain_loss.backward()
        self.optimizer1.step()
        self.zero_gradients()



        domain_pred = self.model(x, step = 2)
        domain_loss = self.w2*self.criterion_CEL(domain_pred, domain_labels)
        domain_loss.backward()
        self.optimizer2.step()
        self.zero_gradients()


        confuse_category = self.model(x, step = 3)
        confuse_category_loss = -self.w2*self.criterion_EL(confuse_category)*self.alpha
        confuse_category_loss.backward()
        self.optimizer3.step()
        self.zero_gradients()

        Fg, Rfg = self.model(x, step = 4) #return recostructor features (Rfg) and extracted features Fg
        reconstructor_loss = self.w3*self.criterion_L2L(Rfg, Fg)
        reconstructor_loss.backward()
        self.optimizer4.step()

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