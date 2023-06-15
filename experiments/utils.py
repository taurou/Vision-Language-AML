import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input):
        p = torch.softmax(input, dim=1)
        log_p = torch.log_softmax(input, dim=1)
        loss = -torch.sum(p * log_p, dim=1).mean()
        return loss
    
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, f_star, f):
        L2loss = torch.norm(f_star - f)
        return L2loss
    
    
def plotValidation(x,y,opt ):
    plt.figure()
    plt.title("Validation accuracy")
    colours=['b']
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000])  
    plt.xlim([0, 5000])
    plt.ylim(0,100)
    plt.plot(x, y, label="valAcc", color=colours[0])
    plt.xlabel("Iterations")   
    #plt.legend([ "Validation acc" ])
      
    plt.savefig('%s/valAccuracy.png' % opt['output_path'], dpi=250 )
    plt.close()

