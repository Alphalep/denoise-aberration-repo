import torch
import math
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal,misc
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2,fftshift,fft,ifft2,ifftshift
from empatches import EMPatches
import imgviz  # just for plotting
import numpy as np
from skimage import color, data, restoration


N = 128# Kernel Size
radius = N//4#np.sqrt(N//np.pi) #Radius of the aperture
center = [N//2,N//2]
lam = 0.55 * 10**-6
wave_number = 2*np.pi/lam
W_d = 0 
W_sa = 0
W_coma= 5 * lam
W_astig = 0
W_field = 0
W_distort  = 0 
#Shift of point in test image
side =2
x_shift = 10
y_shift  = 0
#For patching
patch_size = 64
step_size = 32
overlapPercent = 1- (step_size/patch_size)
#Functions related to Aperture

def get_mag(A):
    ph_mag =  np.sqrt(np.real(A)**2+np.imag(A)**2)
    return ph_mag
def hypergaussian(x,y,radius):
    return(np.exp(-1*((x**2+y**2)/radius**2)**50))

def lcs(img):
    return(img-np.min(img)/(np.max(img)-np.min(img)))

def amplitude_aperture(N,radius,center):
    x = np.linspace(0,N-1,N,dtype = int)
    y = x
    X = x-center[0]
    Y = y-center[1]
    out = np.zeros((N,N))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            out[i,j] = hypergaussian(x,y,radius)

    return out

#General Useful Image Processing algorithms

#FFT Transform
def fft_image(img):
    return(fftshift(fft2(fftshift(img))))
#Display image
def display(img):
    plt.figure()
    plt.axis('off')
    plt.imshow(img,cmap='gray')
    return 0
#Remove 2*pi spikes 
def remove_spikes(img,amp):
    out = img

    out[np.where(img*np.where(amp>0,0,1)>=np.pi)]=0.0
    return out

#General Expansion of a complex valued 2d function

def expand_complex_image(phase_mask):
    ph_real = np.real(phase_mask)
    ph_imag = np.imag(phase_mask)
    ph_mag =  np.sqrt(ph_imag**2+ph_real**2)
    ph_phase = np.arctan2(ph_imag,ph_real)
        
    return(ph_real,ph_imag,ph_mag,ph_phase)



#Normalization of the window function in case of rectangular windows
def generate_norm_factor(shape,patch_size,step_size):
    img = np.ones(shape,dtype = float)
    overlapPercent = 1-(step_size/patch_size)
    emp = EMPatches()
    img_patches, indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlapPercent)
    window = np.zeros(img.shape,float)
    sum = np.zeros(img.shape,dtype=float)
    for id,patch in enumerate(img_patches):
        ind = indices[id]
        window[ind[0]:ind[1],ind[2]:ind[3]]= 1
        temp = img*window
        sum = sum + temp
        window[ind[0]:ind[1],ind[2]:ind[3]]= 0
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            sum[i,j] = 1/sum[i,j]
    return(sum)
#Creating a dotted pattern of image over black background
def star_field(img_size,step_size):
    img = np.zeros(img_size,float)
    for i in range(0,img.shape[0],step_size):
        for j in range(0,img.shape[1],step_size):
            img[i,j]= 1.0
            img[i-1,j-1]=1.0
            img[i+1,j+1]=1.0
            img[i+1,j-1]=1.0
            img[i-1,j+1]=1.0
            img[i,j+1]=1.0
            img[i,j-1]=1.0
            img[i+1,j]=1.0
            img[i-1,j]=1.0
    return img

#Seidel Aberration Function 
def seidel (p0,q0,x,y):
    beta = np.arctan2(q0,p0)
    h2 = np.sqrt(p0**2 + q0**2)
    #rotation of grid
    xr = x*np.cos(beta)+y*np.sin(beta)
    yr = -x*np.sin(beta) + y*np.cos(beta)

    #Seidel Aberration function

    rho2 = xr**2 + yr**2

    W = W_d*rho2 + W_sa*rho2**2 + W_coma*h2*rho2*xr + W_astig*h2**2*xr**2 + W_field*h2**2*rho2 + W_distort*h2**3*xr
    return W

#Phase of the Aperture function 
def phase_aperture(N,center,radius,phase_constant,p0,q0):
    x = np.linspace(0,N-1,N,dtype = int)
    y = x
    X = x-center[0]
    Y = y-center[1]
    X = X/radius
    Y = Y/radius
    out = np.zeros((N,N),dtype=complex)
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            out[i,j] = np.exp(-1*phase_constant*seidel(p0,q0,x,y)*1j)

    return out


#Window Function#
def tri(patch_size):
    X = np.array(range(0,patch_size))
    Y = np.array(range(0,patch_size))
    X = (X-patch_size//2) 
    Y = (Y-patch_size//2)
    out = np.zeros((patch_size,patch_size),dtype=float)
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            out[i,j]=((1-np.abs(2*x/(patch_size)))*(1-np.abs(2*y/(patch_size))))
    return out

