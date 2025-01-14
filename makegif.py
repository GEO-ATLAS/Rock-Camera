from PIL import Image
import imageio
import os
import numpy as np
dataSetName="PBSet1"
dataType="FixStep05"
version=2
input_dir = f'./tests{version}/{dataType}/{dataSetName}/'

images = []
start=20
end=80
for i in range(start,end):
    image_file=input_dir+f"{i}.jpg"
    img = Image.open(image_file)
    images.append(img)
output_gif = f'./images/{dataType}_{dataSetName}_{start}_{end}_{version}.gif'
# Create the GIF
with imageio.get_writer(output_gif, mode='I', fps=10) as writer:
    for img in images:
        writer.append_data(np.array(img))

print(f'GIF saved as {output_gif}')
