import os
import argparse, sys
#from multiprocessing import Pool
from pathlib import Path
import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

def eliminate_close_points_kdtree(arr, threshold):
    """
    arr: shape (N, D)
    threshold: distance threshold
    """
    chosen = []
    # We’ll build the tree as we go
    for point in arr:
        if len(chosen) == 0:
            chosen.append(point)
        else:
            tree = cKDTree(np.array(chosen))
            # Query neighbors within threshold
            idx = tree.query_ball_point(point, r=threshold)
            if len(idx) == 0:
                chosen.append(point)
    return np.array(chosen)

def print_latex(pai, dora, kiri):
    global start_number
    print(f'\\tekiri{{{start_number}}}{{{pai}}}{{{dora}}}{{{kiri}}}{{}}')

    start_number += 1


parser=argparse.ArgumentParser()
parser.add_argument("--input", help="input image file")
#parser.add_argument("--caption", help="caption")
parser.add_argument("--start-number", help="start of latex numbering")
args=parser.parse_args()
global start_number
start_number = int(args.start_number)
print('\\begin{figure}[h]')
print('\\caption{}')
print('\\label{}')



imgcolor = cv.imread(args.input, cv.IMREAD_COLOR)
imgcolorcopy = imgcolor.copy()
img = cv.cvtColor(imgcolor, cv.COLOR_BGR2GRAY)
assert img is not None, "file could not be read, check with os.path.exists()"
imgcopy = img.copy()
template = cv.imread('needle.png', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
mask = template.copy()
mask[::] = 1
mask[8:h-8,7:w-7] = 0

methods = ['cv.TM_CCORR_NORMED']#, 'TM_SQDIFF']


def match_tile(tile_file, img):
    tile = cv.imread(tile_file, cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, tile, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val, max_loc



method = cv.TM_CCORR_NORMED

# Apply template Matching
res = cv.matchTemplate(img,template,method, None, mask)
threshold = 0.95
loc = np.where( res >= threshold)

processed = 0

count = 0
te = []
pai = ''
l = np.stack((loc[1], loc[0]), axis=-1)
l = eliminate_close_points_kdtree(l, 10)
indices = np.arange(len(l))
sorted_indices = sorted(indices, key=lambda i: (l[i, 1] // 10, l[i, 0]))
l_sorted = l[sorted_indices]
print_separator = False
for pt in l_sorted:
    if print_separator:
        print('\\par\\bigskip')
        print_separator = False
    #if(not os.path.exists(f'{i}.png')):
        #print(f'write {processed}.png')
    cv.imwrite(f'{processed}.png', imgcolor[pt[1]+10:pt[1]+h-10, pt[0]+6:pt[0]+w-6] )
    cv.rectangle(imgcolorcopy, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    max_confidence = 0
    for tile in os.listdir('tiles'):
        confidence, loc = match_tile(f'tiles/{tile}', imgcolor[pt[1]-3:pt[1]+h+3, pt[0]-3:pt[0]+w+3])
        if confidence > max_confidence:
            max_confidence = confidence
            max_loc = loc
            max_tile = tile                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      #print(f'{tile}: {confidence}')

    tilename = Path(max_tile).stem
    te.append(tilename)
    count += 1
    cv.putText(imgcolorcopy, tilename, (pt[0], pt[1]), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
    processed += 1
    if count % 16 == 0:
        #print('te:')
        group = {'m': [], 'p': [], 's': [], 'z': []}
        for color in ['m', 'p', 's', 'z']:
            for i in range(0, 14):
                if(te[i][1] == color):
                    group[color].append(te[i][0])
        for color in ['m', 'p', 's', 'z']:
            if(len(group[color]) > 0):
                pai += ''.join(group[color]) + color
                #print(''.join(group[color]) + color, end='')
        #print()
        #print('dora:'+te[14])
        dora = te[14]
        #print('kiri:' + te[15])
        kiri = te[15]
        count = 0
        te = []

        print_latex(pai, dora, kiri)
        print_separator = True
        pai = ''


print('\\end{figure}')



# plt.subplot(121),plt.imshow(res,cmap='gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(cv.cvtColor(imgcolorcopy, cv.COLOR_BGR2RGB))
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(method)
# plt.show()