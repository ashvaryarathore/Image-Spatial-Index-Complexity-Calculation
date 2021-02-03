import cv2
import numpy as np
import xlsxwriter
import os


#/***************************
# Convolution Calculation
#****************************
def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        # print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    # print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    # print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output
#/******************************
# Convolution Calculation: END
#*******************************


#/*******************************
# Magnitude of Image Calculation
#********************************
def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude
#/*************************************
# Magnitude of Image Calculation: END
#**************************************


#/**************************************
#  SI Calculation
#  Calculating Complexity of the images 
#  Images converted into feature maps.
#***************************************
def SICal(features):

    shp = features.shape
    print(shp)
       
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    si_total = []
    si_ft = sobel_edge_detection(features, filter, verbose=False)
    si_ft = np.array(si_ft)
    
    #we will be using the mean of SI 
    #Studies suggest the mean works best. 
    mean_si_ft = np.mean(si_ft)

    #return mean
    return mean_si_ft #[si_value, np.mean(si_total)]
#/**************************************
#  SI Calculation: End
#***************************************  


#/*****************************************
#  Main calling: 
#  We call SI for each image in the dataset
#****************************************** 

#we will use a common workbook to save the SI calculation
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('SIValues.xlsx')
worksheet = workbook.add_worksheet()


#code to read all the subdirectories
dir = 'C:/Users/rathoraa/Desktop/Onedrive/DataSets/Images/8m-3Images'
list = []
for root, dirs, files in os.walk(dir):
    for name in files:
        list.append(os.path.join(root, name))

#Row and column for writing to a file.
# Start from the first cell. Rows and columns are zero indexed.
    row = 0

for i in list:
    #print(i)
    #write this to the file 
    col = 0
    image = i
    image = cv2.imread(image)
    SIVal = SICal(image)
    
    worksheet.write(row, col, i)
    worksheet.write(row, col+1, SIVal)
    row = row + 1
    print(SIVal)

#*********************************************
#                Main END 
#*********************************************






