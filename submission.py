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



#Exercise1



def shift_phi(phi_array, n, shift_points=64):
    shifted_phi = np.roll(phi_array, n * shift_points)  #Loop translation n * T=n * 64 points(1/64 resolution)
    return shifted_phi
#get_cmn_db
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


def  get_cmn_db_2(shift_number: int, polynomial_order: int):
    t = np.arange(0, 2048)
    #phi_dual = get_daubechies_dual_basis(wavelet_name='db4', level=6)
    db4 = pywt.Wavelet('db4')
    phi, _, _ = db4.wavefun(level=6)
    #print(phi)
    phi = np.pad(phi,(0,1599))
    phi_shift = shift_phi(phi, shift_number, shift_points=64)
    phi_shift_dual = phi_shift / 64
    polynomials = t ** polynomial_order              
    c_mn = np.dot(phi_shift_dual, polynomials)               #compute c_mn
    return c_mn, phi_shift

def compute_and_plot_polynomials_db(L=6, max_degree=3):
    wavelet = pywt.Wavelet('db4')  # You can use db2, db4, etc.
    phi, _, _ = wavelet.wavefun(level=L)
    
    # Make phi to the required length
    phi = np.pad(phi, (0, 2048 - len(phi)))
    
    # Define the x-axis for plotting
    x = np.arange(2048)
    
    # Define the maximum degree of the polynomial
    for degree in range(max_degree+1):
        polynomial = x ** degree  # Create polynomial of the given degree
        signal = np.zeros_like(x, dtype=float) # Initialize the signal (reconstructed polynomial)
        plt.figure(figsize=(10, 6))# Plot the shifted kernels and compute the polynomial coefficients
        for shift_number in range(32 - L):
            coefficient,phi_shift= get_cmn_db_2(shift_number, degree)
            signal += coefficient * phi_shift                                              # Reconstruct the signal (polynomial)
            plt.plot(x, coefficient*phi_shift, label=f'Shifted kernel at n={shift_number}', alpha=0.6) # Plot the shifted kernel
        # Plot the reproduced polynomial
        plt.plot(x, signal, color='black', linewidth=2, label="Reconstructed polynomial")
        plt.title(f"Polynomial of Degree {degree} Reconstructed with Shifted Kernels", fontsize=14)
        plt.xlabel("Index (x)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
#compute_and_plot_polynomials_db(L=7, max_degree=3)





#Exercise2




#get_cmn_bspline
def  get_cmn_bspline(shift_number: int, polynomial_order: int):
    t = np.arange(0, 2048)
    b0 = np.ones(64)#0 order
    b1 = convolve(b0, b0, mode='full')  #1 order
    b2 = convolve(b0, b1, mode='full') #2 order
    b3 = convolve(b0, b2, mode='full') #3 order
    
    phi = np.pad(b3,(0,1795))
    phi_shift = shift_phi(phi, shift_number, shift_points=64)
    wavelets = np.zeros((2048,32))
    #calculate dual basis
    for j in range(32):
        wavelets[:, j] = shift_phi(phi, j, shift_points=64)
    gram_matrix = np.dot(wavelets.T, wavelets)
    gram_matrix_inv = np.linalg.inv(gram_matrix)
    dual_wavelets = np.dot(gram_matrix_inv, wavelets.T)
    # shift dual basis
    phi_shift_dual  = dual_wavelets[shift_number,:]
    polynomials = t ** polynomial_order
    c_mn = np.dot(phi_shift_dual, polynomials)

    return c_mn

def  get_cmn_bspline_2(shift_number: int, polynomial_order: int):
    t = np.arange(0, 2048)
    b0 = np.ones(64)
    b1 = convolve(b0, b0, mode='full')  
    b2 = convolve(b0, b1, mode='full') 
    b3 = convolve(b0, b2, mode='full') 
    
    phi = np.pad(b3,(0,1795))
    phi_shift = shift_phi(phi, shift_number, shift_points=64)
    wavelets = np.zeros((2048,32))
    for j in range(32):
        wavelets[:, j] = shift_phi(phi, j, shift_points=64)
    gram_matrix = np.dot(wavelets.T, wavelets)
    gram_matrix_inv = np.linalg.inv(gram_matrix)
    dual_wavelets = np.dot(gram_matrix_inv, wavelets.T)
    phi_shift_dual  = dual_wavelets[shift_number,:]
    polynomials = t ** polynomial_order
    c_mn = np.dot(phi_shift_dual, polynomials)

    return c_mn, phi_shift

def compute_and_plot_polynomials_b_spline(L = 6,max_degree=3):
    b0 = np.ones(64)  #0 order
    b1 = convolve(b0, b0, mode='full')  #1 order
    b2 = convolve(b0, b1, mode='full')  #2 order
    b3 = convolve(b0, b2, mode='full')  #3 order
    
    print(len(b3))
    phi = np.pad(b3,(0,1795))  #padding to2048

    x = np.arange(2048)
    
    # Define the maximum degree of the polynomial
    for degree in range(max_degree+1):
        polynomial = x ** degree  # Create polynomial of the given degree
        signal = np.zeros_like(x, dtype=float)# Initialize the signal (reconstructed polynomial)
        plt.figure(figsize=(10, 6))# Plot the shifted kernels and compute the polynomial coefficients
        for shift_number in range(32 - L):
            coefficient,phi_shift= get_cmn_bspline_2(shift_number, degree)
            signal += coefficient * phi_shift                                              # Reconstruct the signal (polynomial)
            plt.plot(x, coefficient*phi_shift, label=f'Shifted kernel at n={shift_number}', alpha=0.6) # Plot the shifted kernel
        # Plot the reproduced polynomial
        plt.plot(x, signal, color='black', linewidth=2, label="Reconstructed polynomial")
        plt.title(f"Polynomial of Degree {degree} Reconstructed with Shifted Kernels", fontsize=14)
        plt.xlabel("Index (x)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

#compute_and_plot_polynomials_b_spline(L=4, max_degree=3)





#Exercise3




#annihilating_filter
def annihilating_filter(signal_moments ,K):
    metrix_signal_moments = np.zeros((K,K))  #initialization
    for i in range(K):
        for j in range(K):
            metrix_signal_moments[i,j] = signal_moments[K-1-j+i] #build left matrix
    solution_vector = np.zeros((K))
    for i in range(K):
        solution_vector[i] = -signal_moments[K+i] #build the right vector
    h = np.dot(np.linalg.pinv(metrix_signal_moments),solution_vector)  #caculate by S*H = -s' 
    h = np.hstack(([1], h)) #add the first H
    return h

#innovation_params
def innovation_params(h,K,signal_moments):
    roots = np.roots(h)  # tk is the root of the filter
    tk = roots
    solution_metrix = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            solution_metrix[i,j] = tk[j]**i  #compute left matrix
    solution_metrix = np.linalg.inv(solution_metrix)
    solution_vector = signal_moments[0:K]#compute right vector
    ak = np.dot(solution_metrix, solution_vector)#inner product
    #tk = np.round(tk).astype(int)    #if plot, use this code
    matrix_a_t = np.array([ak, tk])  
    return matrix_a_t


data = loadmat('C:\\Users\\zjy16\\Desktop\\coursework24\\coursework24\\project_files_&_data\\tau.mat')
tau = data['tau']  
tau = tau.reshape(-1)
K = 2
#h = annihilating_filter(tau ,K)
#a_t = innovation_params(h,K, tau)
#print("##a_t###")
#print(a_t)



#Exercise4



def sample_and_reconstruction(L=7, K=2):
    diracs = np.zeros(2048)
    diracs[700] = 200  
    diracs[1200] = 100  #initalization diracs

    db4 = pywt.Wavelet('db4')
    phi, _, _ = db4.wavefun(level=6)  #use db4
    phi = np.pad(phi, (0, 1599))  
    y = np.zeros(32 - L)

    for i in range(32 - L):
        phi_shift = shift_phi(phi, i, shift_points=64)
        y[i] = np.dot(phi_shift, diracs)
    tau_m = np.zeros(2 * K)
    #calculate tau_m
    for i in range(2 * K):
        for j in range(32 - L):
            t = np.arange(2048, dtype=np.float64)
            phi_shift = shift_phi(phi, j, shift_points=64)
            #dual basis
            phi_shift_dual = phi_shift / 64.0 
            polynomials = t ** i
            #inner product
            c_mn = np.dot(phi_shift_dual, polynomials)
            mm = c_mn * y[j]   
            tau_m[i] = tau_m[i] + mm     
    #calculate H           
    h = annihilating_filter(tau_m, K)
    #calculate a_k and t_k   
    a_t = innovation_params(h, K, tau_m)

    # # Plotting
    # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    # diracs1 = np.zeros(2048)
    # a1 = a_t[0,0]
    # a2 = a_t[0,1]
    # t1 =int(a_t[1,0])
    # t2 = int(a_t[1,1])
    # diracs1[t1] = a1
    # diracs1[t2] = a2
    # # Plot Diracs
    # axs[0].stem(diracs, linefmt='C0-', markerfmt='C0o', basefmt=" ")
    # axs[0].set_title('Diracs_groudtruth')
    # axs[0].set_xlabel('Index')
    # axs[0].set_ylabel('Amplitude')

    # # Plot a_t
    # axs[1].stem(y, linefmt='C1-', markerfmt='C1o', basefmt=" ")
    # axs[1].set_title('Sampled y[n]')
    # axs[1].set_xlabel('Index')
    # axs[1].set_ylabel('Amplitude')

    # plt.tight_layout()
    # plt.show()

    return a_t


#reconstrution_a_t = sample_and_reconstruction(7,K = 2)
#print(reconstrution_a_t)






#Exercise5




y = loadmat('C:\\Users\\zjy16\\Desktop\\coursework24\\coursework24\\project_files_&_data\\samples.mat')
data_y = y['y_sampled']

def sample_and_reconstrution_with_y(L=0, K=2, data_y = data_y ):
    #same as exercise4
    db4 = pywt.Wavelet('db4')
    phi, _, _ = db4.wavefun(level=6)
    phi = np.pad(phi, (0, 1599))  
    y = data_y.reshape(-1)
    tau_m = np.zeros(2 * K)
    for i in range(2 * K):
        for j in range(32 - L):
            t = np.arange(2048,dtype=np.float64)
            phi_shift = shift_phi(phi, j, shift_points=64)
            phi_shift_dual = phi_shift / 64.0 
            polynomials = t ** i
            c_mn = np.dot(phi_shift_dual, polynomials)
            mm = c_mn * y[j]
            tau_m[i] = tau_m[i] + mm
    #calculate H 
    h = annihilating_filter(tau_m, K)
    #calculate a t
    a_t = innovation_params(h, K, tau_m)
    return a_t, y
#reconstrution_a_t, sampeled_result = sample_and_reconstrution_with_y(6,K = 2,data_y=data_y )
#print("****")
#print(reconstrution_a_t)





#Exercise6




def sample_and_reconstrution_add_noise(L=6, K=2):
    diracs = np.zeros(2048)
    diracs[700] = 200
    diracs[1200] = 100
    #noise = np.random.randn(N)
    db5 = pywt.Wavelet('db5')
    phi, _, _ = db5.wavefun(level=6)  #use db5 
    print(len(phi))
    phi = np.pad(phi, (0,1471))  #1343,1215
    y = np.zeros(32 - L)
    #set guassian noise
    guassian_mean = 0
    guassian_mean_var = 0
    guassian_std_var = np.sqrt(guassian_mean_var)
    noise = np.random.normal(loc=guassian_mean, scale=guassian_std_var, size=2*K+1)
    for i in range(32 - L):
        phi_shift = shift_phi(phi, i, shift_points=64)
        y[i] = np.dot(phi_shift, diracs)
    tau_m = np.zeros(2 * K+1)
    #generate tau with no noise
    for i in range(2 * K+1):
        for j in range(32 - L):
            t = np.arange(2048,dtype=np.float64)
            phi_shift = shift_phi(phi, j, shift_points=64)
            phi_shift_dual = phi_shift / 64.0 
            polynomials = t ** i
            c_mn = np.dot(phi_shift_dual, polynomials)
            mm = c_mn * y[j]
            tau_m[i] = tau_m[i] + mm
    #add noise to tau
    tau_m = tau_m + noise
    h = annihilating_filter(tau_m, K)
    a_t = innovation_params(h, K, tau_m)
        # Plotting
    #fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    #diracs1 = np.zeros(2048)
    #a1 = a_t[0,0]
    #a2 = a_t[0,1]
    #t1 =int(a_t[1,0])
    #t2 = int(a_t[1,1])
    #diracs1[t1] = a1
    #diracs1[t2] = a2
    # Plot Diracs
    #axs[0].stem(diracs, linefmt='C0-', markerfmt='C0o', basefmt=" ")
    #axs[0].set_title('Diracs_groudtruth')
    #axs[0].set_xlabel('Index')
    #axs[0].set_ylabel('Amplitude')

    # Plot a_t
    #axs[1].stem(diracs1, linefmt='C1-', markerfmt='C1o', basefmt=" ")
    #axs[1].set_title('Annihilating Filter a_t')
    #axs[1].set_xlabel('Index')
    #axs[1].set_ylabel('Amplitude')

    #plt.tight_layout()
    #plt.show()
    return a_t, y,tau_m
a_t, y,tau_m  = sample_and_reconstrution_add_noise(L=9, K=2)
print("-----original----")
print(a_t)





#Exercise7



#tls
def tls(signal_moments, K =2):
    #inital S
    S = np.zeros((K+1,K+1))
    N= 2*K+1
    #generate S
    for i in range(K+1):
        for j in range(K+1):
            S[i,j] = signal_moments[K-j+i]
    STS = np.dot(S.T, S)                                    #generate S.tanspose*S
    eigenvalues, eigenvectors = np.linalg.eig(STS)          #solve eigenvalues and eigenvectors
    min_eigenvalue_index = np.argmin(eigenvalues)           #find smallest eigenvalues index
    H = eigenvectors[:, min_eigenvalue_index]               #find h
    a_t_denoise = innovation_params(H, K, signal_moments)   #compute a,t
    return a_t_denoise


a_t_denoise = tls(tau_m, K =2)
print("-----TLS----")
print(a_t_denoise)

#make the matix to be Toeplitz Matrix
def set_elements_to_diagonal_mean(S):
    N = S.shape[0]
    T = np.zeros_like(S)    
    for diag in range(-N + 1, N):
        diag_elements = np.diag(S, k=diag)
        diag_mean = np.mean(diag_elements)
        for i in range(len(diag_elements)):
            row, col = i + max(0, -diag), i + max(0, diag)
            T[row, col] = diag_mean
    return T

#SVD and keep k largest Singular Value
def svd_keep_top_k(S, K):
    U, Sigma, Vt = np.linalg.svd(S, full_matrices=False)
    Sigma[K:] = 0                                         # make the small Singular value = 0
    S_prime = np.dot(U, np.dot(np.diag(Sigma), Vt))        
    return S_prime
#cadzow
def cadzow(signal_moments, K =2,iterations = 100):
    S = np.zeros((K+1,K+1))
    N= 2*K+1
    for i in range(K+1):
        for j in range(K+1):
            S[i,j] = signal_moments[K-j+i]              #generate S
    #iterations
    for i in range(iterations):
        S = svd_keep_top_k(S, K)                        # SVD and keep k largest Singular Value
        S = set_elements_to_diagonal_mean(S)            # make the matix to be Toeplitz Matrix:calculate the mean of diagonal of the matrix. set all the value to the mean
    #TLS
    STS = np.dot(S.T, S)
    eigenvalues, eigenvectors = np.linalg.eig(STS)
    min_eigenvalue_index = np.argmin(eigenvalues)
    H = eigenvectors[:, min_eigenvalue_index]  
    print("*****")
    print(H)
    a_t_denoise = innovation_params(H, K, signal_moments)
    return a_t_denoise

print("-----Cadzow----")
a_t_denoise = cadzow(tau_m, K =2,iterations = 100)
print(a_t_denoise)


#Exercise8


#get_barycenters
def get_barycenters():
    lr_ims1 = [load_lr_image(k) for k in range(1, 40 + 1)]
    #hr_ref_im = np.array(Image.open('project_files_&_data/HR_Tiger_01.tif')) / 255
    #set threshold
    threshold = 0.27  # Choose an appropriate threshold
    lr_ims = pywt.threshold(lr_ims1, threshold, mode='hard')  # (40,64,64,3)
    #load data
    matdict = loadmat('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/PolynomialReproduction_coef.mat')
    coef_0_0 = matdict['Coef_0_0']
    coef_0_1 = matdict['Coef_0_1']
    coef_1_0 = matdict['Coef_1_0']
    barycenters = np.zeros((40,3,2))
    for k in range(40):
        for l in range(3):
            image = lr_ims[k,:,:,l]

            #calculate geometric moments
            m_00 = np.sum(np.multiply(image, coef_0_0))    
            m_01 = np.sum(np.multiply(image, coef_0_1))
            m_10 = np.sum(np.multiply(image, coef_1_0))

            #calculate xi and yi
            x_i = m_10/m_00
            y_i = m_01/m_00
            if (k == 0):
                x_1=x_i
                y_1=y_i
            x_shift = x_i-x_1  #x_1 = 0
            y_shift = y_i-y_1  #y_1 = 0

            #calculate output
            barycenters[k,l,0] = x_shift
            barycenters[k,l,1] = y_shift

    return barycenters[:,:,0],barycenters[:,:,1]

#the following code is the 8(2) and 8(3)

def get_barycenters_with_threshold():
    #add threshold
    image_list = np.zeros((40,64,64,3))
    lr_ims1 = [load_lr_image(k) for k in range(1, 40 + 1)]
    sum_threshold = 0
    for k in range(1, 40 + 1):
        # load images
        lr_k = load_lr_image(k)
        # process per channel
        for c in range(lr_k.shape[2]):  #process RGB
            median_value = np.median(lr_k[:,:,c])
            mad = np.median(np.abs(lr_k[:,:,c] - median_value))
            sigma = mad / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(lr_k[:,:,c].size))
            sum_threshold = threshold + sum_threshold
            #image_list[k-1,:,:,c] = pywt.threshold(lr_k[:,:,c], threshold, mode='hard')
    average_threshold = sum_threshold/(40*3)
    image_list = pywt.threshold(lr_ims1, average_threshold, mode='hard')
    #proceoss like get_barycenters()
    matdict = loadmat('C:/Users/zjy16/Desktop/coursework24/coursework24/project_files_&_data/PolynomialReproduction_coef.mat')
    coef_0_0 = matdict['Coef_0_0']
    coef_0_1 = matdict['Coef_0_1']
    coef_1_0 = matdict['Coef_1_0']
    barycenters = np.zeros((40,3,2))
    for k in range(40):
        for l in range(3):
            image = image_list[k,:,:,l]
            print()
            m_00 = np.sum(np.multiply(image, coef_0_0))
            m_01 = np.sum(np.multiply(image, coef_0_1))
            m_10 = np.sum(np.multiply(image, coef_1_0))
            x_i = m_10/m_00
            y_i = m_01/m_00
            if (k == 0):
                x_1=x_i
                y_1=y_i
            x_shift = x_i-x_1  #x_1 = 0
            y_shift = y_i-y_1  #y_1 = 0
            barycenters[k,l,0] = x_shift
            barycenters[k,l,1] = y_shift
    lr_ims1 = [load_lr_image(k) for k in range(1, 40 + 1)]
    return lr_ims1,barycenters[:,:,0],barycenters[:,:,1]


def edge_detection(gray_image):
    # Using Sobel operator for edge detection
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    return edges

def sharpen_image(img, edges, alpha=0.06):
    # Sharpening processing
    sharpened_img = img + alpha * edges
    sharpened_img = np.clip(sharpened_img, 0, 1) 
    return sharpened_img

def edge_detection_and_sharpening(coeffs,alpha):
    edges = edge_detection(coeffs)
    image = sharpen_image(coeffs, edges, alpha=alpha)
    return image

def wavelet_sharping(img, wavelet='db1', mode='soft', threshold=None):
    #deniose each channel
    if len(img.shape) == 3: 
        denoised_img = np.zeros_like(img)
        
        for c in range(3):
            coeffs = pywt.dwt2(img[:, :, c], wavelet)  # Wavelet decomposition
            if threshold is None:
                threshold = np.sqrt(2 * np.log(img.size)) * np.median(np.abs(coeffs[1][0]))/0.6754  
            cA, (cH, cV, cD) = coeffs
            denoised_cH = edge_detection_and_sharpening(cH,1.3)     #apply_threshold(coeffs, threshold, mode)
            denoised_cV = edge_detection_and_sharpening(cV,1.3)     #apply_threshold(coeffs, threshold, mode)
            denoised_cD = edge_detection_and_sharpening(cD,1.3)     #apply_threshold(coeffs, threshold, mode)
            
            denoised_coeffs = cA, (denoised_cH, denoised_cV, denoised_cD)
            denoised_img[:, :, c] = pywt.idwt2(denoised_coeffs, wavelet)  # wavelet reconstruction
    
    # The restriction range is 0-1
    denoised_img = np.clip(denoised_img, 0, 1)
    return denoised_img
def low_pass_filter(channel, cutoff_frequency):
        rows, cols = channel.shape
        #Fourier transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        
        # build filter
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        radius = int(min(rows, cols) * cutoff_frequency)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        
        # filter
        dft_shift_filtered = dft_shift * mask
        dft_inverse_shift = np.fft.ifftshift(dft_shift_filtered)
        filtered_channel = np.fft.ifft2(dft_inverse_shift)
        return np.abs(filtered_channel)
def edgesharpening(img, cutoff_frequency=0.1, alpha=1.5):

    # Save the processed image
    filtered_img = np.zeros_like(img)
    for c in range(3):  
        filtered_img[:, :, c] = low_pass_filter(img[:, :, c], cutoff_frequency) #low_pass_filter
    
    # Edge detection (using Sobel operator)
    gray_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edges = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
    
    # Sharpening processing: original image+alpha * edge enhancement
    sharpened_img = img + alpha * edges[:, :, np.newaxis]
    sharpened_img = np.clip(sharpened_img, 0, 1)  # The restriction range is 0-1
    
    # Combining low-pass filtering and sharpening results
    result = filtered_img + (sharpened_img - img)  # Reduce aliasing while preserving details
    result = np.clip(result, 0, 1)
    
    return result




lr_ims1,a,b = get_barycenters_with_threshold()

k = image_fusion(lr_ims1,a,b)

k_wave = wavelet_sharping(k, wavelet='db1', mode='soft')
m =  edgesharpening(k_wave, cutoff_frequency=0.1, alpha=0.33)
l = edgesharpening(k_wave, cutoff_frequency=0.1, alpha=0.1)
denoised_img = np.zeros_like(m)
for c in range(3):
    coeffs1 = pywt.dwt2(m[:, :, c], 'db1')  # Wavelet decomposition
    coeffs2 = pywt.dwt2(l[:, :, c], 'db1')  # Wavelet decomposition
    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2
    denoised_coeffs = cA2,(cH1, cV1,cD2)
    denoised_img[:, :, c] = pywt.idwt2(denoised_coeffs,'db1')  
denoised_img = edgesharpening(denoised_img, cutoff_frequency=0.135, alpha=0.06)
psnr = image_comparison(denoised_img)
print(psnr)
# show image
plt.imshow(denoised_img)
plt.title("With Denoising")
plt.axis('off')  
plt.show()



def wavelet_transform_and_visualization(image, wavelet='db1'):
    # If it is an RGB image, convert it to a grayscale image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        gray_image = image
    
    # Wavelet decomposition
    coeffs = pywt.dwt2(gray_image, wavelet) 
    cA, (cH, cV, cD) = coeffs  

    # plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title('Approximation (cA)')
    plt.imshow(cA, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Horizontal Details (cH)')
    plt.imshow(cH, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Vertical Details (cV)')
    plt.imshow(cV, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Diagonal Details (cD)')
    plt.imshow(cD, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
#wavelet_transform_and_visualization(denoised_img, wavelet='db1')