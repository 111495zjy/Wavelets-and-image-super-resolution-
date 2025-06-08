from image_fusion import load_lr_image     
from image_fusion import image_fusion
from image_fusion import image_comparison
from PIL import Image, ImageEnhance
import cv2
import pywt                     #
import numpy as np      
import matplotlib.pyplot as plt        # 
from scipy.signal import freqz
from scipy.signal import convolve
from scipy.fft import fft, ifft, fftshift
from scipy.io import loadmat
from skimage.restoration import denoise_tv_bregman
import matplotlib.pyplot as plt
import skimage.io
import csv
from skimage import restoration
def shift_phi(phi_array, n, shift_points=64):
    shifted_phi = np.roll(phi_array, n * shift_points)  #Loop translation n * T=n * 64 points(1/64 resolution)
    return shifted_phi


def  get_cmn_db(shift_number: int, polynomial_order: int):
    t = np.arange(0, 2048)
    db4 = pywt.Wavelet('db4')   
    phi, _, _ = db4.wavefun(level=6)                         #scaling function ,db4, number of iter=6
    #print(phi)
    phi = np.pad(phi,(0,1599))                               #padding to 2048
    phi_shift = shift_phi(phi, shift_number, shift_points=64)#generate different phi with translation
    phi_shift_dual = phi_shift / 64                          # dual basis is itself
    polynomials = t ** polynomial_order              
    c_mn = np.dot(phi_shift_dual, polynomials)               #compute c_mn with inner production
    return c_mn