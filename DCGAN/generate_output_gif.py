import glob
from PIL import Image

RESIZE_FACTOR = 0.4
OUTPUT_FILENAME = 'Predictions.gif'
DURATION_PER_FRAME = 150

output_images = glob.glob('Outputs/Pred_Epoch_*.png')
output_images.sort()

img_data = []

for filepath in output_images:
    img = Image.open(filepath)
    w, h = img.size
    w, h = int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)
    img = img.resize((w, h))
    img_data.append(img)

img_data[0].save(
    OUTPUT_FILENAME, 
    save_all = True, 
    append_images = img_data[1:], 
    optimize = False, 
    duration = DURATION_PER_FRAME, 
    loop = 1
)