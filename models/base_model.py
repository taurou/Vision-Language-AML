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
    def __init__(self, opt):
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
        if not opt["dom_gen"]:
            self.domain_classifier = nn.Linear(512, 2) #We consider 2 domains at the time. Source and target domain for the unsupervised learning
        else:
            self.domain_classifier = nn.Linear(512, 3) #We consider the 3 source domains.

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


    def forward(self, x):
        Fg = self.feature_extractor(x) #extracted features, also called Fg in the paper

        Fcs = self.category_encoder(Fg)    #Fcs - Category Specific features
        Fds = self.domain_encoder(Fg)      #Fds - Domain Specific features

        #Category disentanglement
        #1st step(0): Train the category classifier 
        Cc = self.category_classifier(Fcs)  #Category encoded features + Category Classifier
        
        Cd = self.domain_classifier(Fds) 

        #2nd step(1): confuse the (already trained) domain classifier
        Ccd = self.domain_classifier(Fcs)   #Category encoded features + Domain Classifier - Predicted (fooled domain predictor) domains

        #Domain disentanglement
        #1st step(2): Train the domain predictor    
        #Cd = self.domain_classifier(Fds)    #Domain encoded features + Domain Classifier
        
        #2nd step(3): confuse the (already trained) category classifier
        Cdc = self.category_classifier(Fds)  #Category encoded features + Category Classifier - #Predicted (fooled category predictor) Categories
        #Feature Reconstructor(4) - Reconstructing Fg from the Fcs and Fdc (category and domain specific features)
        #Passing the concatenated features of category and domain along the columns to the reconstructor.
        Rfg = self.reconstructor(cat((Fcs, Fds), 1)) #Passing the concatenated features of category and domain along the columns to the reconstructor.
        
        return (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds)



class ClipDisentangleModel(nn.Module):
    def __init__(self, opt):
        super(ClipDisentangleModel, self).__init__()
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
        if not opt["dom_gen"]:
            self.domain_classifier = nn.Linear(512, 2) #We consider 2 domains at the time. Source and target domain for the unsupervised learning
        else:
            self.domain_classifier = nn.Linear(512, 3) #We consider the 3 source domains.

        self.category_classifier = nn.Linear(512, 7) #Just like the base model, we consider 7 categories

        self.CLIP_fullyconnected = nn.Linear(512, 512)

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


    def forward(self, x, clip_features = False):
        Fg = self.feature_extractor(x) #extracted features, also called Fg in the paper

        Fcs = self.category_encoder(Fg)    #Fcs - Category Specific features
        Fds = self.domain_encoder(Fg)      #Fds - Domain Specific features

        #Category and domain disentanglement
        #Train the category classifier 
        Cc = self.category_classifier(Fcs)  #Category encoded features + Category Classifier
        #Train the domain classifier 
        Cd = self.domain_classifier(Fds) 

        #confuse the domain classifier
        Ccd = self.domain_classifier(Fcs)   #Category encoded features + Domain Classifier - Predicted (fooled domain predictor) domains
     
        #confuse the category classifier
        Cdc = self.category_classifier(Fds)  #Category encoded features + Category Classifier - #Predicted (fooled category predictor) Categories

        #Feature Reconstructor - Reconstructing Fg from the Fcs and Fdc (category and domain specific features)
        #Passing the concatenated features of category and domain along the columns to the reconstructor.
        Rfg = self.reconstructor(cat((Fcs, Fds), 1)) #Passing the concatenated features of category and domain along the columns to the reconstructor.
        
        if clip_features is not False: 
            Cf = self.CLIP_fullyconnected(clip_features) #Clip features passing through a fully connected layer
        else:
            Cf = False

        return (Fg, Cc, Cd, Ccd, Cdc, Rfg, Fds, Cf)