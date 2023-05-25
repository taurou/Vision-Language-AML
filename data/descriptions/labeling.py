from PIL import Image
import matplotlib.pyplot as plt
import math

def label_art(domain_path):
    img_path = "./data/PACS/kfold/"

    with open(domain_path, "r") as art_file:
        art_lines = list(map(lambda l: l.split()[0], art_file.readlines()))

    start, end= 0, math.floor(len(art_lines)*1/2)                                           #Jacopo
    #start, end= math.ceil(len(art_lines)*1/2), len(art_lines)                              #Gabri


    for i, l in enumerate(art_lines[start:end]):
        index= str(i+start+1)
        img = Image.open(img_path+l)
        plt.imshow(img)
        plt.title(l)
        plt.suptitle(index)
        print("Labeling", index, l)
        plt.show()
    
    return



#domain_path= "./assigned_label/art_painting_to_label.txt"
#domain_path= "./assigned_label/cartoon_to_label.txt"
#domain_path= "./assigned_label/photo_to_label.txt"
domain_path= "./assigned_label/sketch_to_label.txt"




label_art(domain_path)
input("\nEnd.")
