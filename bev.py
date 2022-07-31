import math
import random
import numpy as np
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
import cv2
from lidar import *

TOP_Y_MIN=-20   
TOP_Y_MAX=+20
TOP_X_MIN=-20
TOP_X_MAX=+20     
TOP_Z_MIN=-2.0   
TOP_Z_MAX= 0.4

TOP_X_STEP=0.1 
TOP_Y_STEP=0.1
TOP_Z_STEP=0.4

def lidar_to_top(lidar):

    idx = np.where (lidar['x']>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['x']<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar['y']>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['y']<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar['z']>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar['z']<TOP_Z_MAX)
    lidar = lidar[idx]

    x = lidar['x']
    y = lidar['y']
    z = lidar['z']
    r = lidar['intensity']
    qxs=((x-TOP_X_MIN)//TOP_X_STEP).astype(np.int32)
    qys=((y-TOP_Y_MIN)//TOP_Y_STEP).astype(np.int32)
    qzs=((z-TOP_Z_MIN)//TOP_Z_STEP).astype(np.int32)
    quantized = np.dstack((qxs,qys,qzs,r)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_STEP)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_STEP)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_STEP)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2
    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(width,height,channel), dtype=np.float32)


    histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for z in range(Zn):
            iz = np.where (quantized[:,2]==z)
            quantized_z = quantized[iz]

            for y in range(Yn):
                iy  = np.where (quantized_z[:,1]==y)
                quantized_zy = quantized_z[iy]

                for x in range(Xn):
                    ix  = np.where (quantized_zy[:,0]==x)
                    quantized_zyx = quantized_zy[ix]
                    if len(quantized_zyx)>0:
                        yy,xx,zz = -x,-y, z

                        #height per slice
                        max_height = max(0,np.max(quantized_zyx[:,2])-TOP_Z_MIN)
                        top[yy,xx,zz]=max_height

                        #intensity
                        max_intensity = np.max(quantized_zyx[:,3])
                        top[yy,xx,Zn]=max_intensity

                        #density
                        count = len(idx)
                        top[yy,xx,Zn+1]+=count

                    pass
                pass
            pass

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(16)

    if 0:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        #top_image = np.clip(top_image,0,255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 1: #unprocess
        top_image = np.zeros((height,width),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y   = qxs[n],qys[n]
            yy,xx = -x,-y
            top_image[yy,xx] += 1

        max_value = np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)



    return top, top_image

lidar = np.load("/root/share/project/didi/data/didi/didi-2/Data/1/15/lidar/1530509304325762000.npy")
top, top_img = lidar_to_top(lidar)
cv2.imwrite("./output/top.png",top_img)
from IPython.display import Image
Image(filename="./output/top.png")