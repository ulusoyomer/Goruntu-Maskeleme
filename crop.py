import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import random
colorsArray = []

def visualize_colors(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        colorsArray.append(color)
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

def changeHsv(b,g,r):
    color = np.uint8([[[int(b),int(g),int(r)]]])
    hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    h = hsv_color[0][0][0]
    s = hsv_color[0][0][1]
    v = hsv_color[0][0][2]
    colorHSV = [h,s,v]
    return colorHSV

def setMask(h,s,v,hsv):
    lowerColor = np.array([h-3,s-45,v-45])
    
    upperColor = np.array([h+3,s+45,v+45])
    mask = cv2.inRange(hsv, lowerColor, upperColor)
    return mask

cropping = False 
x_start, y_start, x_end, y_end = 0, 0, 0, 0
image = cv2.imread(sys.argv[1])
image2 = cv2.imread(sys.argv[1])
i = image.copy()
oriImage = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))
cluster = KMeans(n_clusters=15).fit(reshape)
visualize = visualize_colors(cluster, cluster.cluster_centers_)
visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
cv2.imshow('visualize', visualize)

def mouse_crop(event, x, y, flags, param):   
    global x_start, y_start, x_end, y_end, cropping
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:           
            x_end, y_end = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:        
        x_end, y_end = x, y
        cropping = False  
        refPoint = [(x_start, y_start), (x_end, y_end)]
        
        if len(refPoint) == 2:            
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            img = roi.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h=[]
            k = 0
            
            for i in colorsArray:
                h.append(changeHsv(colorsArray[k][2],colorsArray[k][1],colorsArray[k][0]))
                k += 1
                
            k = 0
            masks = []
            
            for i in h:
                masks.append(setMask(i[0],i[1],i[2],hsv))           
            mask = masks[len(masks)-1] + masks[len(masks)-2] + masks[len(masks)-3] + masks[len(masks)-4]+ masks[len(masks)-5]
            res = cv2.bitwise_and(img,img, mask= mask)
            test = np.argwhere(res == 0)
            
            for i in test:
                r = random.randint(0, 2)
                if (r == 0):
                    res[i[0],i[1]] = [colorsArray[len(colorsArray)-1][2],colorsArray[len(colorsArray)-1][1],colorsArray[len(colorsArray)-1][0]]
                elif (r == 1):
                    res[i[0],i[1]] = [colorsArray[len(colorsArray)-2][2],colorsArray[len(colorsArray)-2][1],colorsArray[len(colorsArray)-2][0]]
                else:
                    res[i[0],i[1]] = [colorsArray[len(colorsArray)-3][2],colorsArray[len(colorsArray)-3][1],colorsArray[len(colorsArray)-3][0]]
                    
            image2[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]] = res
            cv2.imshow('Maskesiz Resim', img)
            cv2.imshow('Sonuc', image2)
            cv2.imshow('Secilen Kisim', mask)
            cv2.imshow('Renk Degisimi', res)
            
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

if not cropping:
    cv2.imshow("image", image2) 
elif cropping:
    cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    
cv2.waitKey(0) 
cv2.destroyAllWindows()
