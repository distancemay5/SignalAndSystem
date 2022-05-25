import cv2
import numpy as np
import utilsv5 as utils

def main():
    utils.init_folds()
    filename = './source/lena.png'
    taget_filename = './source/pic.jpg'
    pic = utils.rgb2gray(filename , taget_filename)

    spectrum , shifted_spectrum = utils.dft(pic)
    spectrum_pic = utils.get_spectrum_as_picture(shifted_spectrum)
    cv2.imwrite("./result/spectrum.jpg" , spectrum_pic)

    filter_range = 30
    high_pass_filtered , mask = utils.high_pass_filtering(filter_range , pic.shape , shifted_spectrum)
    cv2.imwrite("./result/high_pass/high_pass_filtered.jpg" , high_pass_filtered)

    low_pass_filtered , mask1 = utils.low_pass_filtering(filter_range , pic.shape , shifted_spectrum)
    cv2.imwrite("./result/low_pass/low_pass_filtered.jpg" , low_pass_filtered)

    range1 = 30
    range2 = 10
    range_y = None
    band_pass_filtered , mask2 = utils.band_pass_filtering(range1 , range2 , pic.shape , shifted_spectrum)
    cv2.imwrite("./result/band_pass/band_pass_filtered.jpg" , band_pass_filtered)

    rank = 1
    butterworth_lp_filtered , mask3 = utils.butterworth_filtering(rank , filter_range , pic.shape , shifted_spectrum , "low_pass")
    cv2.imwrite("./result/butterworth/butterworth_filtered.jpg" , butterworth_lp_filtered)
    
    butterworth_hp_filtered , mask4 = utils.butterworth_filtering(rank , filter_range , pic.shape , shifted_spectrum , "high_pass")
    cv2.imwrite("./result/butterworth/butterworth_hp_filtered.jpg" , butterworth_hp_filtered)
    
    new_spectrum = utils.get_spectrum_as_picture(utils.dft(high_pass_filtered)[1])
    cv2.imwrite("./result/high_pass/high_pass_filtered_spectrum.jpg" , new_spectrum)

    new_spectrum1 = utils.get_spectrum_as_picture(utils.dft(low_pass_filtered)[1])
    cv2.imwrite("./result/low_pass/low_pass_filtered_spectrum.jpg" , new_spectrum1)

    new_spectrum2 = utils.get_spectrum_as_picture(utils.dft(band_pass_filtered)[1])
    cv2.imwrite("./result/band_pass/band_pass_filtered_spectrum.jpg" , new_spectrum2)

    new_spectrum3 = utils.get_spectrum_as_picture(utils.dft(butterworth_lp_filtered)[1])
    cv2.imwrite("./result/butterworth/butterworth_filtered_spectrum.jpg" , new_spectrum3)

    new_spectrum4 = utils.get_spectrum_as_picture(utils.dft(butterworth_hp_filtered)[1])
    cv2.imwrite("./result/butterworth/butterworth_hp_filtered_spectrum.jpg" , new_spectrum4)
    
    H = np.fft.fftshift(utils.degradation_function(pic , a = 0, b = 0.1 , T = 1))
    blurred_pic = utils.motion_blur_v2(pic , H)
    cv2.imwrite("./result/motion_blur/motion_blured.jpg" , blurred_pic)
    spectrum , shifted_spectrum = utils.dft(blurred_pic)
    spectrum_pic_saved = utils.get_spectrum_as_picture(shifted_spectrum)
    cv2.imwrite("./result/motion_blur/blured_spectrum.jpg" , spectrum_pic_saved)

    deblured_pic = utils.inverse_filtering(blurred_pic, H, 0.01)
    cv2.imwrite("./result/motion_blur/motion_deblured.jpg" , deblured_pic)

    blurred_add_noise_pic = utils.add_gauss_noise(blurred_pic , 0.01)
    cv2.imwrite("./result/motion_blur/blurred_add_noise_pic.jpg" , blurred_add_noise_pic)
    spectrum , shifted_spectrum = utils.dft(blurred_add_noise_pic)
    spectrum_pic = utils.get_spectrum_as_picture(shifted_spectrum)
    cv2.imwrite("./result/motion_blur/blured_noised_spectrum.jpg" , spectrum_pic)

    cv2.imwrite("blured_spectrum.jpg" , spectrum_pic)
    try_deblured_with_noise_pic = utils.inverse_filtering(blurred_add_noise_pic, H , 0.01)
    cv2.imwrite("./result/motion_blur/try_deblured_with_noise_pic.jpg" , try_deblured_with_noise_pic)
    deblured_with_noise_pic = utils.wiener_filtering(blurred_add_noise_pic , H , 0.05 , 0.01)
    cv2.imwrite("./result/motion_blur/deblured_with_noise_pic.jpg" , deblured_with_noise_pic)
    
    gauss_blur = cv2.GaussianBlur(blurred_add_noise_pic,(5,5),0)
    cv2.imwrite("./result/motion_blur/gauss_blur.jpg" , gauss_blur)
    spectrum , shifted_spectrum = utils.dft(gauss_blur)
    spectrum_pic = utils.get_spectrum_as_picture(shifted_spectrum)
    cv2.imwrite("./result/motion_blur/gauss_blur_spectrum.jpg" , spectrum_pic)

    utils.show_line_in_plt(pic.shape , utils.get_spectrum_as_picture(shifted_spectrum) , "./result/motion_blur/chart2.jpg")
    utils.show_line_in_plt(pic.shape , spectrum_pic_saved , "./result/motion_blur/chart.jpg")
if __name__ == '__main__':  
    main()