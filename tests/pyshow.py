import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Display.')
parser.add_argument('--file', type=str)
parser.add_argument('--vmin', type=float)
parser.add_argument('--vmax', type=float)
args = parser.parse_args()

img = np.array(Image.open(args.file))
plt.imshow(img, vmin=args.vmin, vmax=args.vmax)
plt.colorbar()
plt.show()
