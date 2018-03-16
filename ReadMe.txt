The folder contains: 

Source code for the project where, 

Question 1: LinearScaling.py
Quetsion 2: HistogramQuantization.py 

Also, two input files are present fruits.jpg and image.bmp 

1. To run the programs, please install python 3.6.4 

2. Go to the folder containing these commands

3. On the terminal please type: (for case OpenCv is not present on the system) 
	pip install opencv-python

4. Type the following commands to run the python code: 
 py <file-name>.py w1 w2 h1 h2 <input-image> <output-image> 

For example: 

For Question 1: py LinearScaling.py 0.2 0.2 0.5 0.7 fruits.jpg outfruits.jpg 

For Question2: py HistogramQuantization.py 0.2 0.2 0.5 0.7 image.bmp outimage.bmp

Explanation: 
1. To handle division by 0 in the program we consider: 
uPrime= 0
vPrime = 0
    if (d = 0):
        # To avoid devision by 0
    	uPrime = float(4*X)/(d+0.00001)
        vPrime = float(9*Y)/(d+0.00001)

    return L, u, v 

2. The above program does not work for monochromatic images i.e gray-scale images and makes it look bad. 
For exam the image gray.jpg in the folder on LinearScaling doesnot change the image output. On HistogramQuantization of the image makes it high contrast as each s-gray level has more number of pixels now. 