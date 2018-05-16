import numpy as np
import cv2
import math

def openimage(img_name):
    return cv2.imread('/home/chiarotti/Documentos/ImageManipulation/images/'+img_name)

def tograyscale(img_src):
    img_blank = np.ones((img_src.shape[0], img_src.shape[1], 1), np.uint8)

    for width in range(img_src.shape[0]):
        for height in range(img_src.shape[1]):
            result = 0
            for channel in range(img_src.shape[2]):
                result = result + img_src.item(width, height, channel)

            result = result / img_src.shape[2]

            img_blank.itemset((width,height,0), result)
    return img_blank


def rgbtoycbcr(img_src):
    img_blank = np.ones((img_src.shape[0], img_src.shape[1], 3), np.uint8)

    for width in range(img_src.shape[0]):
        for height in range(img_src.shape[1]):
            R = img_src.item(width, height, 2)
            G = img_src.item(width, height, 1)
            B = img_src.item(width, height, 0)

            y = 16 + ((65.738*R)/256) + ((129.057*G)/256) + ((25.064*B)/256)
            cb = 128 - ((37.945*R)/256) - ((74.494*R)/256) + ((112.439*B)/256) 
            cr = 128 + ((112.439*R)/256) - ((94.154*G)/256) - ((18.285*B)/256)

            img_blank.itemset((width, height,0), y)
            img_blank.itemset((width, height,1), cb)
            img_blank.itemset((width, height, 2), cr)

    return img_blank

def rgbtohsi(img_src):
    img_blank = np.ones((img_src.shape[0], img_src.shape[1], 3), np.uint8)

    for width in range(img_src.shape[0]):
        for height in range(img_src.shape[1]):
            R = img_src.item(width, height, 2)
            G = img_src.item(width, height, 1)
            B = img_src.item(width, height, 0)

            rn = R / float((R+G+B))
            gn = G / float((R+G+B))
            bn = B / float((R+G+B))

            i = (R+G+B) / float(3*255)
            s = 1.0 - (3 * (min([rn, gn, bn])))
            h = 0
            if(s!=0.0):
                h = math.acos((0.5 * ((rn - gn) + (rn - bn))) / (math.sqrt((rn - gn) * (rn - gn) + (rn - bn) * (gn - bn))))
                if(bn>gn):
                    h = (2 * 3.14159265) -    h

            img_blank.itemset((width,height,0),(h*180)/3.14159265359)
            img_blank.itemset((width, height, 1),s*100)
            img_blank.itemset((width, height, 2), i*255)

    return img_blank


def limiar(img_src):
    img_grayscale = tograyscale(img_src)
    lista_color = {}
    for pos in range(0,256):
        lista_color[pos] = 0

    for width in range(img_grayscale.shape[0]):
        for height in range(img_grayscale.shape[1]):
            color = img_grayscale.item(width,height,0)
            lista_color[color] +=1
            
    
    max_val = 0
    max_pos = 0
    for pos in range(0,256):
        if(lista_color[pos] > max_val):
            max_val = lista_color[pos]
            max_pos = pos

    img_blank = np.ones((img_grayscale.shape[0], img_grayscale.shape[1], 1), np.uint8)
    for width in range(img_grayscale.shape[0]):
        for height in range(img_grayscale.shape[1]):
            if(img_grayscale.item(width,height,0) <= max_pos):
                img_blank.itemset((width,height,0), 0)
            else:
                img_blank.itemset((width,height,0), 255)
    return img_blank


def mediansmoothing(img_src, size):
    width_limit = img_src.shape[0]
    height_limit = img_src.shape[1]

    color_list = []
    pixel_matrix = []
    img_grayscale = tograyscale(img_src)
    img_blank = np.ones((img_grayscale.shape[0], img_grayscale.shape[1], 1), np.uint8)

    for width in range(width_limit):
        for height in range(height_limit):
            pixel_matrix = getNeighbors(width,height,width_limit,height_limit,size)
            for w,h in pixel_matrix:
                if(w and h != -1):
                    color_list.append(img_grayscale[w][h][0])

            color_list.sort()
            if(len(color_list) % 2 == 0):
                res = (int((color_list[(len(color_list)/2)-1])) + int((color_list[(len(color_list)/2)]) )) / 2
            else:
                res = color_list[len(color_list)/2]

            img_blank.itemset((width,height,0),res)
            color_list = []

    return img_blank

def average(img_src, mask_num):
    mask = []
    if(mask_num == 0):
            mask = setmask([0.2, [[0,1,0],[1,1,1],[0,1,0]]])
    elif(mask_num == 1):
            mask = setmask([0.11, [[1,1,1],[1,1,1],[1,1,1]]])
    elif(mask_num == 2):
            mask = setmask([0.1, [[1,1,1],[1,2,1],[1,1,1]]])
    elif(mask_num == 3):
            mask = setmask([0.08, [[1,2,1],[2,4,2],[1,2,1]]])

    width_limit = img_src.shape[0]
    height_limit = img_src.shape[1]

    img_grayscale = tograyscale(img_src)
    img_blank = np.ones((img_grayscale.shape[0], img_grayscale.shape[1], 1), np.uint8)    

    for width in range(width_limit):
        for height in range(height_limit):
            pixel_matrix = getNeighbors(width,height,width_limit,height_limit, 3)
            res_final = 0
            aux = 0
            for w,h in pixel_matrix:
                if(w and h != -1):
                    res_final += img_grayscale[w][h][0] * mask[aux]
                    aux+=1
                    img_blank.itemset((width,height,0), res_final)    

    return img_blank

def convolution(img_src, size):
    laplacian_mask = laplacemask(size)
    img_grayscale = tograyscale(img_src)
    img_blank = np.ones((img_grayscale.shape[0], img_grayscale.shape[1], 1), np.uint8)    
    
    for width in range(img_src.shape[0]):
        for height in range(img_src.shape[1]):
            pixel_matrix = getNeighbors(width, height, img_src.shape[0], img_src.shape[1], size)

            res = 0
            aux = 0
            for w,h in pixel_matrix:
                if(w and h != -1):
                    res += int(img_grayscale[w][h][0]) * int(laplacian_mask[aux])

                aux+=1

            if(res < 0):
                res = 0
            img_blank.itemset((width,height,0), res)
    return img_blank

def setmask(mask):
    new_mask = []
    for line in  mask[1]:
        for element in line:
            new_mask.append(mask[0]*element)
    return new_mask

def getNeighbors(width, height, width_limit, height_limit, size):
    matrix = []
    for w in range((-1*((size/2))+width), ((size/2)+1)+width):
        for h in range((-1*((size/2))+height), ((size/2)+1)+height):
            if(w >= 0 and w < width_limit):
                if(h >= 0 and h < height_limit):
                    matrix.append([w,h])
                else:
                    matrix.append([-1,-1])
            else:
                matrix.append([-1,-1])
    
    return matrix    

def laplacemask(size):
    laplace = []
    for pos in range(size*size):
        if(pos == (size*size)/2):
            laplace.append(((size*size)-1))
        else:
            laplace.append(-1)
    
    return laplace