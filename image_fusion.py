from scipy.io import loadmat
from scipy import interpolate

import numpy as np
from numpy.fft import fft2,ifft2,fft,ifft,rfft2,irfft2
import pywt

import matplotlib.pyplot as plt

from PIL import Image


def load_lr_image(k):
    lr_k = np.array(Image.open('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/LR_Tiger_{:02d}.tif'.format(k)))
    return lr_k/255

n_sensors = 40
n_layers = 3

lr_ims = [load_lr_image(k) for k in range(1,n_sensors+1)]
hr_ref_im = np.array(Image.open('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/HR_Tiger_01.tif'))/255

# Get repro coeffs
matdict = loadmat('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/PolynomialReproduction_coef.mat')
coef_0_0 = matdict['Coef_0_0']
coef_0_1 = matdict['Coef_0_1']
coef_1_0 = matdict['Coef_1_0']


hr_size = 512
lr_size = 64
scale = int(hr_size/lr_size)

hr_samples_col,hr_samples_row = np.meshgrid(range(0,hr_size),range(0,hr_size))

lr_samples_col = np.tile(range(0,hr_size-1,scale),(lr_size,1))
lr_samples_row = np.transpose(lr_samples_col)
spline = loadmat('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/2DBspline.mat')['Spline2D']


######################################################################
# Internal functions for image fusion
######################################################################

def image_fusion(ims,txs,tys):
    interp_vals,interp_pts = zip(*[get_interpolation_points(im,tx,ty) for im,tx,ty in zip(ims,txs,tys)])
    interp_vals = np.concatenate(interp_vals,axis=1)
    interp_pts = np.concatenate(interp_pts,axis=0)
    
    query_points = list(zip(hr_samples_col.flatten(),hr_samples_row.flatten()))

    layers = []
    for ilayer in range(0,3):
        # Interpolation of 2D scattered data
        data_image = interpolate.griddata(interp_pts[:,:,ilayer],interp_vals[ilayer,:],query_points,'cubic',fill_value=0.)
        data_image = data_image * (scale**2)
        layers = layers + [data_image]
    
    psf = spline
    
    sr = [ layer.reshape((hr_size,hr_size)) for layer in layers]
    sr = np.stack(sr,axis=2)
    sr = edgetaper(sr,psf)
    sr = deconvwnr(sr,psf)
    sr = np.clip(sr,a_min=0,a_max=None)/np.max(sr,axis=(0,1))
  
    return sr

def get_interpolation_points(im,tx,ty):
    im = im.transpose((2,0,1)) # Put layer first
    im = im.reshape((3,lr_size*lr_size))
    shifted_cols = np.expand_dims(lr_samples_col.flatten(),axis=1) - np.expand_dims(tx,axis=0)
    shifted_rows = np.expand_dims(lr_samples_row.flatten(),axis=1) - np.expand_dims(ty,axis=0)

    interpolation_pts = np.stack([shifted_cols,shifted_rows],axis=1)
    interpolation_vals = im
    
    return interpolation_vals,interpolation_pts

def psf2otf(psf,shape):
    deltaW = shape[0] - psf.shape[0]
    deltaL = shape[1] - psf.shape[1]
    psf_padded = np.pad(psf,((0,deltaW),(0,deltaL)))
    psf_padded = np.roll(psf_padded,(-(psf.shape[0]//2) , -(psf.shape[1]//2)), (0,1))
    
    f = rfft2(psf_padded)
    return f

def edgetaper(im,psf):
    def padded_acf(x,pad_to):
        power_spect = np.square(np.abs(fft(x,pad_to)))
        acf = np.real(ifft(power_spect))
        acf = np.append(acf[1:],acf[-1])/np.max(acf)  #Not sure
        return acf
    #normalise psf
    psf = psf / np.sum(psf)
    
    # Get psf projections
    projections = [np.sum(psf,axis=ax) for ax in range(0,psf.ndim)]
    acfs = [padded_acf(proj,sh) for proj,sh in zip(projections,im.shape)]
    
    weighting = np.expand_dims(1-acfs[0],axis=0) * np.expand_dims(1-acfs[1],axis=1)
    weighting = np.expand_dims(weighting,axis=2)
    
    otf = psf2otf(psf,im.shape)
    otf = np.expand_dims(otf,axis=2)

    blurred = irfft2(rfft2(im,axes=(0,1)) * otf ,axes=(0,1))
    mix = weighting*im + (1-weighting)*blurred    
    
    clipped = np.clip(mix,a_max=np.max(im),a_min=np.min(im))
    return clipped

def deconvwnr(im,psf):
    # Mostly from matlab deconvwnr    
    otf = psf2otf(psf,im.shape)
    otf = np.expand_dims(otf,axis=2)

    denom = np.square(np.abs(otf))
    numer = np.conj(otf) * rfft2(im,axes=(0,1))
   
    eps = np.finfo(float).eps

    tiny = np.max(np.abs(numer))*np.sqrt(eps)*2

    tiny_inds = np.abs(denom) < tiny
    tiny_signs = 2*(np.real(denom[tiny_inds]) > 0) - 1

    denom[tiny_inds] = tiny_signs * tiny

    jft = numer / denom
    filtered = irfft2(jft,axes=(0,1))
    
    return filtered

def image_comparison(sr):
    hr_ref = hr_ref_im
    lr_ref = lr_ims[0]
    
    mse = (np.square(hr_ref - sr)).mean()
    psnr = 10 * np.log10(1/mse)
   
    fig , axes = plt.subplots(1,3)
    axes[0].imshow(lr_ref, interpolation="nearest")
    axes[0].set_xlabel("Low resolution")
    axes[1].imshow(sr, interpolation="nearest")
    axes[1].set_xlabel("Super resolved")
    axes[2].imshow(hr_ref, interpolation="nearest")
    axes[2].set_xlabel("High resolution")
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(top=0.85)
    plt.suptitle("PSNR: %f" % psnr,y=-0.0)
    plt.show()

    return psnr


######################################################################
# Function to test image fusion
######################################################################

    
def superres(barycenters_f):
    sr = image_fusion(lr_ims, *barycenters_f())
    return image_comparison(sr) , sr
