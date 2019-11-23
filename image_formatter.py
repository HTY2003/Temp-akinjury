import os
from os import listdir
from os.path import isfile, join
import cv2
from PIL import Image

def main(paths):
    avg_width = 0
    avg_height = 0

    #read files
    onlyfiles = []
    for path in paths:
        onlyfiles += [path+f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.jpg') or f.endswith('.png')or f.endswith('.jpeg')]

    #find average dimensions
    data = {}
    data['images_count'] = len(onlyfiles)
    for filename in onlyfiles:
        im = Image.open(filename)
        width, height = im.size
        avg_width += width
        avg_height += height

    data["avg_width"] = round(avg_width/data["images_count"])
    data["avg_height"] = round(avg_height/data["images_count"])
    impt = (data["avg_width"], data["avg_width"])

    #resizing operations
    for i in range(len(onlyfiles)):
        filename = onlyfiles[i]
        im = Image.open(filename)
        width, height = im.size
        im = im.resize(impt)
        filename = "cleaned" + filename
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        im = im.convert("RGB")
        im.save(filename)
    print(data)


if __name__ == '__main__':
    main(['healine/firstdegburn/', \
                'healine/minorcut/', \
                'healine/contusion/', \
                'healine/snakebite/', \
                'healine/nosebleed/'])
