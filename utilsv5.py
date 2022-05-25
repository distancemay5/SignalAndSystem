import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def init_folds():
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists("result/high_pass"):
        os.makedirs("result/high_pass")
    if not os.path.exists("result/low_pass"):
        os.makedirs("result/low_pass")
    if not os.path.exists("result/band_pass"):
        os.makedirs("result/band_pass")
    if not os.path.exists("result/butterworth"):
        os.makedirs("result/butterworth")
    if not os.path.exists("result/motion_blur"):
        os.makedirs("result/motion_blur")

# 生成理想低通、高通、带通滤波器掩膜
def low_pass_filter(shape, radius):
    h , w = shape
    mask = np.zeros((shape[0] , shape[1] , 2), dtype=np.uint8)
    cv2.circle(mask, (int(w / 2), int(h / 2)), radius, (1, 1, 1), -1)
    filter_for_show = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(filter_for_show, (int(w / 2), int(h / 2)), radius, (255, 255, 255), -1)   
    return mask , filter_for_show

def high_pass_filter(shape, radius):
    h , w = shape
    mask = np.ones((shape[0] , shape[1] , 2), dtype=np.uint8)
    cv2.circle(mask, (int(w / 2), int(h / 2)), radius, (0, 0, 0), -1)
    filter_for_show = np.full(shape[:2], 255 , dtype=np.uint8)
    cv2.circle(filter_for_show, (int(w / 2), int(h / 2)), radius, (0, 0, 0), -1)
    return mask , filter_for_show

def band_pass_filter(shape, radius1 , radius2):
    h , w = shape
    mask = np.zeros((shape[0] , shape[1] , 2), dtype=np.uint8)
    cv2.circle(mask, (int(w / 2), int(h / 2)), radius1, (1, 1, 1), -1)
    cv2.circle(mask, (int(w / 2), int(h / 2)), radius2, (0, 0, 0), -1)
    filter_for_show = np.zeros(shape[:2] , dtype=np.uint8)
    cv2.circle(filter_for_show, (int(w / 2), int(h / 2)), radius1, (255, 255, 255), -1)
    cv2.circle(filter_for_show, (int(w / 2), int(h / 2)), radius2, (0, 0, 0), -1)
    return mask , filter_for_show

def butterworth_lp_filter(shape, rank, radius):
    # 中心位置
    h, w = shape[:2]
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    mask = 1 / (1 + np.power(dis / radius, 2 * rank))
    filter_for_show = mask * np.full((shape) , 255 , dtype=np.uint8)
    mask = mask.reshape(shape[0] , shape[1] , 1).repeat(2,axis=2)
    return mask , filter_for_show

def butterworth_hp_filter(shape, rank, radius):
    # 中心位置
    h, w = shape[:2]
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    mask = 1 - 1 / (1 + np.power(dis / radius, 2 * rank))
    filter_for_show = mask * np.full((shape) , 255 , dtype=np.uint8)
    mask = mask.reshape(shape[0] , shape[1] , 1).repeat(2,axis=2)
    return mask , filter_for_show

def rgb2gray(filename , target_filename):
    rgb = cv2.imread(filename)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target_filename , gray)
    return gray

def dft(img):
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32,flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft , dft_shift

def get_spectrum_as_picture(spectrum):
    spectrum = cv2.magnitude(spectrum[:,:,0],spectrum[:,:,1])
    spectrum = np.maximum(spectrum , 1)
    magnitude_spectrum = 20 * np.log(spectrum)
    return magnitude_spectrum

def high_pass_filtering(filter_range , shape , spectrum):
    # 理想高通滤波器: 将中心半径30内的谐波全部滤掉(置为0)
    mask , high_pass_filter_for_show = high_pass_filter(shape , filter_range)
    filtered_spectrum = mask * spectrum
    # 将频谱从中心低频的状态移动回原来的状态
    shifted_back = np.fft.ifftshift(filtered_spectrum)
    # 傅里叶反变换
    img_back = cv2.idft(shifted_back)
    # 计算幅度值
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # 微调,见实验报告
    img_back = np.uint8(img_back / (shape[0] * shape[1]))
    
    show_high_pass = get_spectrum_as_picture(filtered_spectrum)
    cv2.imwrite("./result/high_pass/high_pass_filtered_spectrum1.jpg" , show_high_pass)
    cv2.imwrite("./result/high_pass/high_pass_filter.jpg" , high_pass_filter_for_show)
    return img_back , mask

def low_pass_filtering(filter_range , shape , spectrum):
    # 理想低通滤波器: 只保留中心方圆整数30内的谐波(置为1)
    mask , low_pass_filter_for_show = low_pass_filter(shape , filter_range)
    filtered_spectrum = mask * spectrum
    # 将频谱从中心低频的状态移动回原来的状态
    shifted_back = np.fft.ifftshift(filtered_spectrum)
    # 傅里叶反变换
    img_back = cv2.idft(shifted_back)
    # 计算幅度值
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # 微调,见实验报告
    img_back = np.uint8(img_back / (shape[0] * shape[1]))
    
    show_low_pass = get_spectrum_as_picture(filtered_spectrum)
    cv2.imwrite("./result/low_pass/low_pass_filtered_spectrum1.jpg" , show_low_pass)
    cv2.imwrite("./result/low_pass/low_pass_filter.jpg" , low_pass_filter_for_show)
    return img_back , mask

def band_pass_filtering(range1 , range2 , shape , spectrum):
    # 理想带通滤波器: 只保留range范围内的谐波
    mask , band_pass_filter_for_show = band_pass_filter(shape , range1 , range2)
    filtered_spectrum = mask * spectrum
    # 将频谱从中心低频的状态移动回原来的状态
    shifted_back = np.fft.ifftshift(filtered_spectrum)
    # 傅里叶反变换
    img_back = cv2.idft(shifted_back)
    # 计算幅度值
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # 微调,见实验报告
    img_back = np.uint8(img_back / (shape[0] * shape[1]))

    show_band_pass = get_spectrum_as_picture(filtered_spectrum)
    cv2.imwrite("./result/band_pass/band_pass_filtered_spectrum1.jpg" , show_band_pass)
    cv2.imwrite("./result/band_pass/band_pass_filter.jpg" , band_pass_filter_for_show)
    return img_back , mask

def butterworth_filtering(rank , filter_range , shape , spectrum , mode):
    if mode == 'low_pass':
        mask , butterworth_filter_for_show = butterworth_lp_filter(shape , rank , filter_range)
    else:
        mask , butterworth_filter_for_show = butterworth_hp_filter(shape , rank , filter_range)
    filtered_spectrum = mask * spectrum
    # 将频谱从中心低频的状态移动回原来的状态
    shifted_back = np.fft.ifftshift(filtered_spectrum)
    # 傅里叶反变换
    img_back = cv2.idft(shifted_back)
    # 计算幅度值
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # 微调,见实验报告本处注释
    img_back = np.uint8(img_back / (shape[0] * shape[1]))
    
    show_low_pass = get_spectrum_as_picture(filtered_spectrum)
    if mode == "low_pass":
        cv2.imwrite("./result/butterworth/butterworth_filtered_spectrum1.jpg" , show_low_pass)
        cv2.imwrite("./result/butterworth/butterworth_filter.jpg" , butterworth_filter_for_show)
    else:
        cv2.imwrite("./result/butterworth/butterworth_hp_filtered_spectrum1.jpg" , show_low_pass)
        cv2.imwrite("./result/butterworth/butterworth_hp_filter.jpg" , butterworth_filter_for_show)       
    return img_back , mask

def inverse_filtering(input, H, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = H + eps # 避免除数为0添加的一个极小量
    result = np.fft.ifft2(input_fft / PSF_fft) 
    result = np.abs(result)
    return result

def add_gauss_noise(img, sigma):
    img = img/255
    noise = np.random.normal(0,sigma,img.shape)
    output = img + noise
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output

def wiener_filtering(input_signal, H, K , eps):
    input_signal_cp = np.copy(input_signal) # 输入信号的副本
    input_signal_cp_fft = np.fft.fft2(input_signal_cp)  # 输入信号的傅里叶变换
    PSF_fft = H + eps 
    h_abs_square = np.abs(PSF_fft)**2 # 退化函数模值的平方
    # 维纳滤波
    output_signal_fft = np.conj(PSF_fft) / (h_abs_square + K)
    output_signal = np.abs(np.fft.ifft2(output_signal_fft * input_signal_cp_fft)) # 输出信号傅里叶反变换
    return output_signal

def degradation_function(pic , a=0, b=0 , T=1):
    [r,c] = pic.shape
    u = np.arange(r).reshape((-1,1)) - np.ceil(r/2)
    v = np.arange(c)-np.ceil(c/2)
    tmp = np.pi*(u*a + v*b)     #广播机制得到矩阵
    tmp[tmp==0]=1e-20
    H = T*np.sin(tmp)/tmp*np.exp(-1j*tmp);  # 退化函数
    return H

def motion_blur_v2(pic , H):
    g = np.fft.ifft2(H * np.fft.fft2(pic));  
    g = np.uint8(np.real(g))
    return g


def show_line_in_plt(shape , spectrum , filename):
    size = spectrum.shape[0]
    sample = [ it.sum()/it.shape[0] for it in spectrum.T]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[0, shape[1]], ylim=[0, 255], title='look for dark line',
       ylabel='brightness', xlabel='position')
    x = np.arange(0, shape[1])
    y = sample
    ax.plot(x, y)
    plt.savefig(filename)
    plt.show()


