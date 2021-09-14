# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import division
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import requests
import argparse
import sys
from efficientnet_pytorch import EfficientNet

from torchvision.transforms.transforms import Grayscale


def progress(count, total, suffix=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def deg2num(lat_deg, lon_deg, zoom):
    """
    latitude et longitude en °, le zoom donne la racine du nombre de tuiles couvrant
    le monde, maximum 18
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


"""Ici on a la fonction download qui automatise le téléchargement"""

def download(path, latitude, longitude, zoom = 17):
    #Lat and lon from N and E

    directory = "download_images"
    for filename in os.listdir(directory):
        os.remove(f"download_images/{filename}")

    i=0
    
    xtile, ytile = deg2num(latitude, longitude, zoom)
    for a in np.linspace(-3,3,7) :     #Pour récupérer 49 images autour des coordonnées données.
        for b in np.linspace(-3,3,7) :
            progress(i,48)

            _x, _y = xtile+b, ytile+a

            url = "https://api.mapbox.com/v4/mapbox.satellite"
            url = url + f"/{zoom}"
            url = url + f"/{_x}/{_y}"
            url = url + "@2x.png256?access_token=pk.eyJ1IjoibWFydHltbm91bW91cyIsImEiOiJja25hNHMzcWwwa2lmMndsY2ljYWxqc3JpIn0.7A91C28dNIV5pe1sH-UM7Q"
            r = requests.get(url)
            it = "0"*(3-len(str(i)))+str(i)  #pour que 10 ne soit pas avant 2 par exemple.
            with open(path + 'image'+it+str(_x)+str(_y)+'.png','wb') as f:
                f.write(r.content)

            i+=1



parser = argparse.ArgumentParser( description="Input long and lat and gives sat img")
parser.add_argument(
    "--long", default="0.0", help="input longitude value"
    )

parser.add_argument(
    "--lat", default="45.0", help="input lattitue"
    )

parser.add_argument(
    "--download_path", default = 'download_images/', help="Image download path"
) 

parser.add_argument(
    "--model", default="squeezenet"
)


args = parser.parse_args()


lat = float(args.lat)
long = float(args.long)
path = args.download_path

print('\n'+"Downloading the satellite image...")
download(path, lat, long)


data_dir = ""
use_pretrained=True
num_classes=2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Chargement du modèle
if args.model == "squeezenet" :
    model_ft_norm = models.squeezenet1_0(pretrained=use_pretrained)
    model_ft_norm.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    model_ft_norm.num_classes = num_classes
    input_size = 224

    model_ft_norm.load_state_dict(torch.load("models/squeezenet5.pth"))

elif args.model == "efficientnet":
    """ Efficientnet
    """
    model_ft_norm = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    output_filter = model_ft_norm._fc.in_features
    model_ft_norm._fc = nn.Linear(output_filter, num_classes)
    input_size = 224

    model_ft_norm.load_state_dict(torch.load("models/efficientnet5.pth"))

else :
    sys.stdout.write('\n'+'The model asked is not valid.'+'\n')
    sys.stdout.flush()

# Transformation appliquée aux images durant l'entraînement
transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.303, 0.348, 0.234], [0.157, 0.112, 0.102]),
        ]
)

print('\n'+"Detecting the windturbines ...")


model_ft_norm.eval()
    

fig = plt.figure(figsize=(8.95,9))
plt.title("REOL - détection d'éoliennes"+'\n', fontsize=20)
plt.axis('off')
fig.patch.set_visible(False)

i=0
directory = "download_images"
for filename in os.listdir(directory):
    progress(i,48)
    img = Image.open(f"download_images/{filename}").convert('RGB')    # On récupère les images
    x = transform(img).unsqueeze(0)                       # On les transforme de la même manière qu'ont été transformées les éoliennes ayant servies à l'entraînement
    output = model_ft_norm(x)                                         # Application du modèle
    prob = round(float(output[0][1]/(output[0][0]+output[0][1])),3)   # Détermination de la probabilité de présence d'au moins une éolienne.
    _, pred = torch.max(output,1)
    ax = fig.add_subplot(7,7,i+1)
    trans = transforms.Grayscale()
    if int(pred)==0 :                      # Comme code couleur, on a choisi de mettre les images sans éoliennes en gris et celles avec en couleur.
        ax.imshow(trans(img), cmap='gray')
    else :
        ax.imshow(img)
        if args.model == "squeezenet" :
            ax.set_title(str(prob), color='r')  # On indique en plus la confiance (probabilité) lorsque le modèle détecte des éoliennes.
    ax.axis('off')

    i+=1

plt.subplots_adjust(wspace=0, hspace=0)

sys.stdout.write('\n'+"Saving the processed image ..."+'\n')
sys.stdout.flush()

plt.savefig('resultat.png', dpi=1000)
plt.clf()

sys.stdout.write("Displaying the final image ...")
sys.stdout.flush()

img_f = plt.imread('resultat.png')
plt.imshow(img_f)
plt.axis('off')
fig.patch.set_visible(False)
plt.show()