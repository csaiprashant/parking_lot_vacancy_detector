# Parses all ground truth xml files and generates positive training images

import xml.etree.ElementTree
import cv2
import glob
import os

a = 1
for filename in glob.glob("PKLot\\*\\sunny\\*\\*.xml"):
    imgname = os.path.splitext(filename)[0] + ".jpg"
    e = xml.etree.ElementTree.parse(filename).getroot()

    coords = dict()
    for space in e.findall('space'):
        ID = space.get('id')
        occupied = space.get('occupied')
        if occupied == '1':
            coords[ID] = []
            for point in space.iter('point'):
                x = point.attrib.get('x')
                y = point.attrib.get('y')
                coords[ID].append((x, y))

    coords = {int(k): v for k, v in coords.items()}

    imgname = os.path.splitext(filename)[0] + ".jpg"
    img = cv2.imread(imgname)

    for i in coords.keys():
        vertices = coords[i]
        x = [int(x) for (x, y) in vertices]
        y = [int(y) for (x, y) in vertices]
        x1 = min(x) - 10
        y1 = min(y) - 10
        x2 = max(x) + 10
        y2 = max(y) + 10
        name = 'positive_samples\\img' + str(a) + '.jpg'
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(name, cropped)
        a = a + 1
