import torch.nn as nn
from torch import cat
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()
        
        if len(x.size()) < 2:
            x = x.unsqueeze(0)
        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 2) #We just consider 2 domains at the time. Source and target domain.
        self.category_classifier = nn.Linear(512, 7) #Just like the base model, we consider 7 categories

        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )


    def forward(self, x, step):
        x = self.feature_extractor(x) #extracted features, also called Fg in the paper
        
        #Category disentanglement
        #1st step(0): Train the category classifier 
        if step == 0: 
            x = self.category_encoder(x)       #Fcs - Category Specific features
            x = self.category_classifier(x)    #Predicted Categories

        #2nd step(1): confuse the (already trained) domain classifier
        elif step == 1:
            x = self.category_encoder(x)       #Fcs - Category Specific features
            x = self.domain_classifier(x)      #Predicted (fooled domain predictor) domains

        #Domain disentanglement
        #1st step(2): Train the domain predictor    
        elif step == 2:
            x = self.domain_encoder(x)          #Fds - Domain Specific features
            x = self.domain_classifier(x)       #Predicted domain
        
        #2nd step(3): confuse the (already trained) category classifier
        elif step == 3:
            x = self.domain_encoder(x)          #Fds - Domain Specific features
            x = self.category_classifier(x)     #Predicted (fooled category predictor) Categories
        
        #Feature Reconstructor(4) - Reconstructing Fg from the Fcs and Fdc (category and domain specific features)
        #Passing the concatenated features of category and domain along the columns to the reconstructor.
        elif step == 4:
            Fcs = self.category_encoder(x)      #Fcs - Category Specific features
            Fds = self.domain_encoder(x)        #Fds - Domain Specific features
            Rfg = self.reconstructor(cat((Fcs, Fds), 1))
            return x, Rfg #return recostructor features (Rfg) and extracted features Fg (x)

        return x


    '''
    def forward(self, x):
        Fg = self.feature_extractor(x) #extracted features, also called Fg in the paper

        Fcs = self.category_encoder(Fg)    #Fcs - Category Specific features
        Fds = self.domain_encoder(Fg)      #Fds - Domain Specific features

        #Cc and Cd = Classify category and domain
        Cc = self.category_classifier(Fcs)  #Category encoded features + Category Classifier
        Cd = self.domain_classifier(Fds)    #Domain encoded features + Domain Classifier
        
        #Cross-Adversarial training - C(enc. features)(classifier)
        Ccd = self.domain_classifier(Fcs)   #Category encoded features + Domain Classifier
        Cdc = self.category_classifier(Fds) #Domain encoded features + Category Classifier

        #Feature Reconstructor - Reconstructing Fg from the Fcs and Fdc (category and domain specific features)
        Rfg = self.reconstructor(cat((Fcs, Fds), 1)) #Passing the concatenated features of category and domain along the columns to the reconstructor.
        return (Fg, Cc, Cd, Ccd, Cdc, Rfg)
    '''    
