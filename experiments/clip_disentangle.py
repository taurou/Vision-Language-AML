import torch
import clip
from experiments.utils import *
from models.base_model import ClipDisentangleModel
class CLIPDisentangleExperiment: # See point 4. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        #Setup clip
        # Load CLIP model and freeze it
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup model
        self.model = ClipDisentangleModel(opt)
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])

        # Optimizers for all parts of the network.
        self.criterion_CEL = torch.nn.CrossEntropyLoss()
        self.criterion_EL = EntropyLoss()
        self.criterion_L2L = L2Loss() 

        # Weights 
        self.w1 = opt["weights"][0]
        self.w2 = opt["weights"][1]
        self.w3 = opt["weights"][2]
        self.alpha = opt["weights"][3]
        self.clip = opt["weights"][4] 
        print("CLIP parameters: \n","w1: ", self.w1, "w2: ", self.w2, "w3: ", self.w3, "alpha: ", self.alpha, "clip: ", self.clip)


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

        self.optimizer.zero_grad()
        if not self.opt['dom_gen']:
            if(targetDomain == False):
                x = data[0]
                y = data[1]
                x = x.to(self.device)
                y = y.to(self.device)           
                domain_labels = torch.zeros(len(x), dtype=torch.long).to(self.device) 
            else:
                x = data[0]
                x = x.to(self.device)
                domain_labels = torch.ones(len(x), dtype=torch.long).to(self.device) 

            if(len(data) > 2 ): #if the data also contains the descriptions.
                descr = data[2]
                tokenized_text = clip.tokenize(descr).to(self.device)
                
                text_features = self.clip_model.encode_text(tokenized_text)
                (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds, Cf) = self.model(x, text_features)
            else:
                (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds, Cf) = self.model(x)

            category_loss = 0 if targetDomain == True else self.criterion_CEL(Cc, y) #TODO rivedere ordine dei parametri
            
            confuse_domain_loss = -self.criterion_EL(Ccd)

            domain_loss = self.criterion_CEL(Cd, domain_labels)

            confuse_category_loss = -self.criterion_EL(Cdc)

            reconstructor_loss = self.criterion_L2L(Rfg, Fg)

            clip_loss = self.criterion_L2L(Fds, Cf) if Cf is not False else 0

            loss = self.w1*(category_loss + self.alpha*confuse_domain_loss) + self.w2*(domain_loss + self.alpha*confuse_category_loss) + self.w3*reconstructor_loss + self.clip*clip_loss
            loss.backward()
            self.optimizer.step()

            return loss.item()
        
        else: #Domain generalization setting
            x = data[0]
            y = data[1]
            domain_labels = data[2]
            x = x.to(self.device)
            y = y.to(self.device)           
            domain_labels = domain_labels.to(self.device) 

            if(len(data) > 3 ): #if the data also contains the descriptions.
                descr = data[3]
                tokenized_text = clip.tokenize(descr).to(self.device)
                text_features = self.clip_model.encode_text(tokenized_text)
                (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds, Cf) = self.model(x, text_features)
            else:
                (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds, Cf) = self.model(x)



            category_loss = self.criterion_CEL(Cc, y) #TODO rivedere ordine dei parametri
            
            confuse_domain_loss = -self.criterion_EL(Ccd)

            domain_loss = self.criterion_CEL(Cd, domain_labels)

            confuse_category_loss = -self.criterion_EL(Cdc)

            reconstructor_loss = self.criterion_L2L(Rfg, Fg)

            clip_loss = self.criterion_L2L(Fds, Cf) if Cf is not False else 0

            loss = self.w1*(category_loss + self.alpha*confuse_domain_loss) + self.w2*(domain_loss + self.alpha*confuse_category_loss) + self.w3*reconstructor_loss + self.clip*clip_loss
            loss.backward()
            self.optimizer.step()

            return loss.item()

        

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for data in loader:
                x = data[0]
                y = data[1]
                x = x.to(self.device)
                y = y.to(self.device)

                (_, Cc,_, _, _, _, _, _) = self.model(x)
                loss += self.criterion_CEL(Cc, y)
                pred = torch.argmax(Cc, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss

    