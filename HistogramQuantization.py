import cv2
import numpy as np
import sys
import math

class Pixel:
    L = 0
    u = 0
    v = 0
    def __init__(self, L, u, v):
        self.L = L
        self.u = u
        self.v = v

# Device independent sRGB: Converting Non-Linear sRGB to Linear RGB
def inverseGamma(v):
    if (v < 0.03928):
        v = v/12.92
    else:
        v = pow(((v+0.055)/1.055), 2.4)
    return v    

# Device independent sRGB: Converting RGB to XYZ 
def RGBtoXYZ(r, g, b):
    convertMatrix = [[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]]
    return np.matmul(convertMatrix, [r, g, b])

# Converting Non-Linear sRGB to Non-Linear RGB, simply divide by 255 
def sRGBtoRGB(n):
    n = float(n/255)
    return n

# COnverting XYZ to RGB 
def XYZtosRGB(X, Y, Z):
    convertMatrix = [[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648, -0.204043, 1.057311]]
    return np.matmul(convertMatrix, [X, Y, Z]) 

# Converting LUV to XYZ 
def LUVtoXYZ(L, u, v):
    Xw, Yw, Zw = 0.95, 1.0, 1.09
    uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    vw = 9*Yw/(Xw + 15*Yw + 3*Zw)
    
    #For L = 0 
    uPrime = 0
    vPrime = 0
    if (L != 0):
        uPrime = (u + 13*uw*L)/(13*L)
        vPrime = (v + 13*vw*L)/(13*L)

    if (L > 7.9996):
        Y = pow(((L+16)/116), 3) * Yw
    else:
        Y = (L/903.3)*Yw

    # for the case v' = 0 
    X = 0
    Z = 0
    if (vPrime != 0):
        X = ( Y * 2.25 * uPrime ) / vPrime
        Z = ( Y * (3 - 0.75*uPrime - 5*vPrime)/ vPrime )

    return X, Y, Z
    
# Converting XYZ to LUV, 6.1 
def XYZtoLUV(X, Y, Z):
    Xw, Yw, Zw = 0.95, 1.0, 1.09
    uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    vw = 9*Yw/(Xw + 15*Yw + 3*Zw)
    #For each pixel in X, Y, Z computing: 
    #For the range when L is: 0 <= L <= 100
    t = Y/Yw
    if (t > 0.008856):
        L = 116*pow(t, (1/3)) - 16
    else:
        L = 903.3*t

    d = X + 15*Y + 3*Z
   
    uPrime = 0
    vPrime = 0
    if (d != 0):
        uPrime = 4*X/d
        vPrime = 9*Y/d
     # To avoid devision by 0
    else: 
        uPrime = float(4*X)/(d+0.00001)
        vPrime = float(9*Y)/(d+0.00001)

    u = 13* L* (uPrime - uw)
    v = 13* L* (vPrime - vw)

    return L, u, v 

# Defining Gamma function for d 
def gamma(d):
    if (d < 0.00304):
        return 12.92*d
    else:
        return (1.055*pow(d, (1/2.4)) - 0.055)

def clip(n):
    if (n < 0):
        n = 0
    elif (n > 1):
        n = 1
    return n
    

if __name__ == '__main__':
    
    if(len(sys.argv) != 7) :
        print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
        print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
        print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
        sys.exit()

    w1 = float(sys.argv[1])
    h1 = float(sys.argv[2])
    w2 = float(sys.argv[3])
    h2 = float(sys.argv[4])
    name_input = sys.argv[5]
    name_output = sys.argv[6]

    if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
        print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
        sys.exit()

    inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
    if(inputImage is None) :
        print(sys.argv[0], ": Failed to read image from: ", name_input)
        sys.exit()

    cv2.imshow("input image: " + name_input, inputImage)

    rows = len(inputImage)
    cols = len(inputImage[0])

    temp  = [[0 for x in range(cols)] for y in range(rows)] 

    for i in range(rows) :
        for j in range(cols) :
            
            b, g, r = inputImage[i, j]
            
            # Non-Linear sRGB to Linear RGB
            r, g, b = sRGBtoRGB(r), sRGBtoRGB(g), sRGBtoRGB(b)
            
            # Non-Linear RGB to Linear RGB 
            r, g, b = inverseGamma(r), inverseGamma(g), inverseGamma(b)

            # Linear RGB to XYZ
            X, Y, Z = RGBtoXYZ(r, g, b)

            # Converting XYZ to LUV 
            L, u, v = XYZtoLUV(X, Y, Z)        
          
            temp[i][j] = Pixel(L, u, v)
           
    
    rows, cols, bands = inputImage.shape 
    W1 = round(w1*(cols-1))
    H1 = round(h1*(rows-1))
    W2 = round(w2*(cols-1))
    H2 = round(h2*(rows-1))

    # Finding frequency 
    frequency = np.zeros(101)
    data = np.zeros(101)

    for i in range(H1, H2) :
        for j in range(W1, W2) :
            L, u, v = temp[i][j].L, temp[i][j].u, temp[i][j].v
            L = int(round(L))
            if(L <= 0):
                frequency[0]= frequency[0] + 1
            elif(L >= 100):
                frequency[100]= frequency[100] + 1
            else:
                frequency[L]= frequency[L] + 1

    for i in range(1, len(frequency)):
        frequency[i] = frequency[i] + frequency[i-1]

    for i in range(0, len(frequency)):
        if(i == 0):
            data[i] = math.floor((frequency[i]/2)*(101/frequency[100]))
        else:
            data[i] = math.floor(((frequency[i]+frequency[i-1])/2)*(101/frequency[100]))    

    for i in range(0, rows) :
        for j in range(0, cols) :
            L = temp[i][j].L
            L = int(round(L))
            if(L < 0):
                L = data[0]
            elif(L > 100):
                L = data[100]
            else:
                L = data[L]
            temp[i][j].L = L

  
    outputImage = np.zeros([rows, cols, bands], dtype = np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            L, u, v = temp[i][j].L, temp[i][j].u, temp[i][j].v

            # Converting LUV to XYZ, calling method LUVtoXYZ()
            X, Y, Z = LUVtoXYZ(L, u, v)

            # Converting XYZ to linear sRGB, calling method XYZtoSRGB()
            r, g, b = XYZtosRGB(X, Y, Z)

            r, g, b = clip(r), clip(g), clip(b)

            # Coverting linear sRGB to Non-linear sRGB, calling methof gamma()
            r, g, b = 255*gamma(r), 255*gamma(g), 255*gamma(b)

            # output image in the form of BGR
            outputImage[i, j] = [b, g, r]

    cv2.imshow("Output", outputImage)
    cv2.imwrite(name_output, outputImage);            

    # wait for key to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()