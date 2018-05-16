import imgprocess

img_name = 'raioX.bmp'
img_src = imgprocess.openimage(img_name)

#img_grayscale = imgprocess.tograyscale(img_src)
#img_ycbcr = imgprocess.rgbtoycbcr(img_src)
#img_hsi = imgprocess.rgbtohsi(img_src)
#img_limiar = imgprocess.limiar(img_src)
img_smoothing = imgprocess.mediansmoothing(img_src,5)
#img_average = imgprocess.average(img_src,3)
img_convolution = imgprocess.convolution(img_smoothing, 11)

imgprocess.cv2.imshow('Original', img_src)
#imgprocess.cv2.imshow('GrayScale',img_grayscale)
#imgprocess.cv2.imshow('YCbCr', img_ycbcr)
#imgprocess.cv2.imshow('HSI', img_hsi)
#imgprocess.cv2.imshow('Limiar', img_limiar)
#imgprocess.cv2.imshow('Smoothing', img_smoothing)
#imgprocess.cv2.imshow('Average', img_average)
imgprocess.cv2.imshow('Laplace', img_convolution)
imgprocess.cv2.waitKey(0)