import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm



def grayscale_raster_creation(input_MSfile, output_filename):
    """ 
    This function creates a grayscale brightness image from an input image to be used for multi - scale - texture index calculation. 
    For every pixel in the input image, the intensity values from the red, green, blue channels are first obtained, and the maximum of 
    these values are then assigned as the pixel's intensity value, which would give the grayscale brightness image as mentioned earlier, 
    as per standard practice in the remote sensing academia and industry. It is assumed that the first three channels of the input 
    image correspond to the red, green and blue channels, irrespective of order.
    
    Inputs:
    - input_MSfile: File path of the input image that needs to be converted to grayscale brightness image
    - output_filename: File path of the grayscale brightness image that is to be written to file
    
    Outputs:
    - gray: Numpy array of grayscale brightness image of corresponding multi - channel input image
    
    """
    
    image = np.transpose(gdal.Open(input_MSfile).ReadAsArray(), [1, 2, 0])
    gray = np.zeros((int(image.shape[0]), int(image.shape[1])))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i, j] = max(image[i, j, 0], image[i, j, 1], image[i, j, 2])
    
    input_dataset = gdal.Open(input_MSfile)
    input_band = input_dataset.GetRasterBand(1)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff_driver.Create(output_filename, input_band.XSize, input_band.YSize, 1, gdal.GDT_Float32)
    output_dataset.SetProjection(input_dataset.GetProjection())
    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.GetRasterBand(1).WriteArray(gray)
    
    output_dataset.FlushCache()
    del output_dataset
    
    return gray

  

def texture_attribute_image(input_gray_filename, output_filename, min_scale, max_scale, step_size, attribute, order, 
                            displ_dist, second_order_attribute, GL = 128):
    """ 
    This function is used to create the texture images of the various first - order and second order multi - scale texture 
    attributes defined in the paper 'Spatial Context - Dependent Multi - Scale and Directional Image Texture' by 
    Timothy A. Warner & Jong Yeol Lee (2011). 
    
    Inputs:
    - input_gray_filename: String or path of input grayscale image to be used.
    - output_filename: String or path of output multi - scale texture image to be written.
    - min_scale: Minimum sliding window size to be applied across the original grayscale image. (must be an odd number).
    - max_scale: Maximum sliding window size to be applied across the original grayscale image. (must be an odd number).
    - step_size: Spatial increment of sliding window size.
    - attribute: Type of multi - scale texture to be calculated (must be one of 
                                                                 ['minimum', 'median', 'mean', 'maximum', 'range']).
    - order: Order of texture to be calculated (must be either 'first' or 'second').
    - displ_dist: Magnitude of displacement vector to be used for calculation of second - order texture.
    - second_order_attribute: Type of second order texture to be calculated (must be one of 
                                                                             ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 
                                                                              'energy', 'correlation'])
    - GL: Maximum gray level to be used for calculation of second - order texture.
    
    Output:
    - attribute_img: Numpy array which represents the calculated attribute image for the input grayscale image.
    
    """

    if (step_size % 2 != 0):
        raise ValueError('Please input multiples of 2 for step_size.')
    
    if (min_scale % 2 == 0):
        raise ValueError('Please input an odd number for min_scale.')
    
    if (max_scale % 2 == 0):
        raise ValueError('Please input an odd number for max_scale.')
        
    if (attribute not in ['minimum', 'median', 'mean', 'maximum', 'range']):
        raise ValueError("Please input either 'minimum' or 'median' or 'mean' or 'maximum' or 'range' for attribute.")
        
    if (order not in ['first', 'second']):
        raise ValueError("Please input either 'first' or 'second' for order.")
        
    if min_scale < displ_dist:
        raise ValueError('Please ensure that min_scale is larger than or equal to displ_dist.')
        
    if (second_order_attribute not in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']):
        raise ValueError("Please input either 'contrast' or 'dissimilarity' or 'homogeneity' or 'ASM' or 'energy' or 'correlation' for second_order_attribute.")
    
    gray_img = gdal.Open(input_gray_filename).ReadAsArray()
    max_buffer = int((max_scale - 1) / 2)
    gray_img_padded = np.zeros(((gray_img.shape[0] + 2 * max_buffer), (gray_img.shape[1] + 2 * max_buffer)))
    gray_img_padded[max_buffer : (max_buffer + gray_img.shape[0]), max_buffer : (max_buffer + gray_img.shape[1])] = gray_img
    
    txt_img = np.zeros((gray_img.shape[0], gray_img.shape[1], int((max_scale - min_scale) / step_size) + 1))
    attribute_img = np.zeros((gray_img.shape[0], gray_img.shape[1]))
    
    if order == 'first':
        for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
            buffer = int((scale - 1) / 2)
            for i in range(max_buffer, gray_img_padded.shape[0] - max_buffer):            
                for j in range(max_buffer, gray_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                    array = gray_img_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                    txt_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = np.var(array)
    elif order == 'second':
        gray_img_rescaled = exposure.rescale_intensity(gray_img_padded, out_range = (0, GL - 1)).astype('uint8')
        for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
            buffer = int((scale - 1) / 2)
            for i in range(max_buffer, gray_img_padded.shape[0] - max_buffer):            
                for j in range(max_buffer, gray_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                    array = gray_img_rescaled[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                    glcm = greycomatrix(array, [displ_dist], [0, np.pi / 4, np.pi / 2, (3 / 4) * np.pi], levels = 128, 
                               normed = True)
                    props = greycoprops(glcm, prop = second_order_attribute)
                    txt_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = props.mean()
                    
                    
    for i in range(attribute_img.shape[0]):
        for j in range(attribute_img.shape[1]):
            
            if (attribute == 'minimum'):
                attribute_img[i, j] = np.min(txt_img[i, j, :])
            elif (attribute == 'median'):
                attribute_img[i, j] = np.median(txt_img[i, j, :])
            elif (attribute == 'mean'):
                attribute_img[i, j] = np.mean(txt_img[i, j, :])
            elif (attribute == 'maximum'):
                attribute_img[i, j] = np.max(txt_img[i, j, :])
            elif (attribute == 'range'):
                attribute_img[i, j] = np.max(txt_img[i, j, :]) - np.min(txt_img[i, j, :])
    
    input_dataset = gdal.Open(input_gray_filename)
    input_band = input_dataset.GetRasterBand(1)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff_driver.Create(output_filename, input_band.XSize, input_band.YSize, 1, gdal.GDT_Float32)
    output_dataset.SetProjection(input_dataset.GetProjection())
    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    output_dataset.GetRasterBand(1).WriteArray(attribute_img)    
    output_dataset.FlushCache()
    output_dataset.GetRasterBand(1).ComputeStatistics(False)
    del output_dataset
    
    return attribute_img
