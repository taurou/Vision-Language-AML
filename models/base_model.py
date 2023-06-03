import torch.nn as nn
from torch import cat
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):  # We use the ResNet18 pretrained on ImageNet as feature extractor, subclass of nn.Module
    def __init__(self):             
        super(FeatureExtractor, self).__init__()        # We call the constructor of the superclass nn.Module
        self.resnet18 = resnet18(pretrained=True)       # We load the pretrained ResNet18 (hyperparameters pretrained on ImageNet)
    
    def forward(self, x):                               # We define the forward pass of the network: the input is the image tensor x
        x = self.resnet18.conv1(x)                      # first convolutional layer
        x = self.resnet18.bn1(x)                        # first batch normalization layer
        x = self.resnet18.relu(x)                       # first ReLU activation
        x = self.resnet18.maxpool(x)                    # max pooling layer
        x = self.resnet18.layer1(x)                      
        x = self.resnet18.layer2(x)                     
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()                                 # We remove the extra dimensions (1, 1) to obtain a tensor of dimension (512)
        
        if len(x.size()) < 2:                           # If the tensor has less than 2 dimensions, we add a dimension in position 0
            x = x.unsqueeze(0)                          # This is necessary because the following operations require a 2D tensor as input
        return x

class BaselineModel(nn.Module):                     # Baseline model is a subclass of nn.Module
    def __init__(self):                                 
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor() # Feature extractor is a ResNet18 pretrained on ImageNet, it extracts features from the input image
        self.category_encoder = nn.Sequential(      # Category encoder is a sequential model made of 3 fully connected layers, in input it takes the output of the feature extractor
            nn.Linear(512, 512),                    # nn.Linear is a fully connected layer: first parameter is the number of input features, second parameter is the number of output features
            nn.BatchNorm1d(512),                    # nn.BatchNorm1d is a batch normalization layer: it normalizes the output of the previous layer
            nn.ReLU(),                              # nn.ReLU is a non-linear activation function

            nn.Linear(512, 512),                    
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)         # It's the final fully connected layer, it takes in input the output of the category encoder and it outputs the class scores
                                                    # it means that it outputs a vector of 7 elements
    
    def forward(self, x):                           # We define the forward pass of the network: the input is the image tensor x
        x = self.feature_extractor(x)               # We extract the features from the input image
        x = self.category_encoder(x)                # We encode the useful features in the category space
        x = self.classifier(x)                      # We classify the image in one of the 7 classes, the output is a vector of 7 elements
        return x                                    # We return the output vector containing the class scores

class DomainDisentangleModel(nn.Module):
    def __init__(self, opt):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor() # As before, we use the ResNet18 pretrained on ImageNet as feature extractor

        self.domain_encoder = nn.Sequential(        # The goal of this encoder is to extract domain specific features, similarly to the category encoder but for the domain
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
            self.domain_classifier = nn.Linear(512, 2) # We consider 2 domains at the time. Source and target domain for the unsupervised learning
        else:
            self.domain_classifier = nn.Linear(512, 3) # We consider the 3 source domains.

        self.category_classifier = nn.Linear(512, 7)   # Just like the base model, we consider 7 categories

        self.reconstructor = nn.Sequential(             # The goal of the reconstructor is to reconstruct the original image from the extracted features
            nn.Linear(1024, 512),                       # it's done to test the quality of the extracted features. If the features are good, the reconstructed image should be similar to the original one
            nn.BatchNorm1d(512),                        # The input of the reconstructor is the concatenation of the category and domain encoded features
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )


    def forward(self, x):
        Fg = self.feature_extractor(x)      # extracted features, also called Fg in the paper

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