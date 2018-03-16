import cv2
import numpy as np
import sys
import math

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

rows, cols, bands = inputImage.shape 
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

# The transformation should be based on the historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

# defining gamma correction method
def gammaCorrection(D):
    if(D<0.00304):
        I =12.92*D
    else:
        I = (1.055*(math.pow(D, (1 / 2.4)))) - 0.055
    return I;

# defining the inverse gamma function 
#Converting (R', G', B') to (R, G, B)
def inverseGamma(v):
    if (v < 0.03928):
       color = v/12.92
    else:
       color = math.pow(((v+0.055)/1.055),2.4)
    return color;

        
# getting the L value for the LUV function
#Computing for each pixel X, Y, Z : XYZ to LUV 6.1 
def gettingLValue(t):
    if (Y>0.008856):
        L = (116 * math.pow(t,1/3)) - 16
    else:
        L = (903.3 * t)
    return L;

#LUV to XYZ 6.2 
# Given L, computing Y 
def gettingYfromL(L,Yw):
    if(L>7.9996):
       Y = math.pow(((L + 16) / 116), 3)*Yw
    else:
       Y = (L / 903.3)*Yw
    return Y;

  
tmp = np.copy(inputImage)
       
# Getting the L minimum and the maximum values for the window 
Lmin = 999.0 #defining the default minimum value 
Lmax = -999.0 #defining the default maximum value 

for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
		# Converting RGB values to Non-Linear RGB
        b = b/255
        g = g/255
        r = r/255
		# Converting Non-Linear to Linear-RGB
        b = inverseGamma(b)
        g = inverseGamma(g)
        r = inverseGamma(r)
		# converting to XYZ using matrix multiplication
        X =  0.412453*r + 0.35758*g + 0.180423*b
        Y =  0.212671*r + 0.71516*g + 0.072169*b
        Z =  0.019334*r + 0.119193*g + 0.950227*b
		#converting to Luv
        L = gettingLValue(Y)
        if(L<Lmin):
             Lmin=L
        if(L>Lmax):
            Lmax=L

difference = Lmax- Lmin
        
for i in range(0,rows) :
    for j in range(0,cols) :
        b, g, r = inputImage[i, j]
        # Converting RGB values to Non-Linear RGB
        b = b/255
        g = g/255
        r = r/255

		# Converting Non-Linear to Linear-RGB
        b = inverseGamma(b)
        g = inverseGamma(g)
        r = inverseGamma(r)

		# converting to XYZ by matrix multiplication
        X =  0.412453*r + 0.35758*g + 0.180423*b
        Y =  0.212671*r + 0.71516*g + 0.072169*b
        Z =  0.019334*r + 0.119193*g + 0.950227*b

		# converting to Luv
        Xw = 0.95
        Yw = 1.0
        Zw = 1.09
        uw = ((4 * Xw) / (Xw + (15 * Yw) + (3 * Zw)))
        vw = ((9 * Yw) / (Xw + (15 * Yw) + (3 * Zw)))
        L = gettingLValue(Y)
        d = X + 15*Y + 3*Z
        if( d == 0.0):
           ud = 0.0
           vd = 0.0
        else:
           ud = ((4 * X) / d)
           vd = ((9 * Y) / d)
        
        u = 13 * (L * (ud - uw))
        v = 13 * (L * (vd - vw))

		# Performing Linear Scaling
        L = ((L - Lmin) * 100) / difference;
		
        # Mapping
        if(L>100):
           L=100
        if(L<0):
           L=0
		# Converting back to XYZ
        if(L !=0):
           ud = ((u + (13 * uw*L)) / (13 * L))
           vd = ((v + (13 * vw*L)) / (13 * L))
        else:
           ud = 0
           vd = 0
        Y = gettingYfromL(L,Yw)
        if(vd == 0):
          X = 0
          Z = 0
        else:
          X = Y*2.25*(ud / vd)
          Z = ((Y*(3 - (0.75*ud) - (5 * vd))) / vd)

		  # To Linear sRGB by matrix multiplication
          Rs = 3.240479*X - 1.53715*Y - 0.498535*Z;
          Gs = -0.969256*X + 1.875991*Y + 0.041556*Z;
          Bs = 0.055648*X - 0.204043*Y + 1.057311*Z;
          
		  #vClipping
          if(Rs < 0):
            Rs = 0
          if(Gs < 0):
            Gs = 0
          if(Bs < 0):
            Bs = 0
          if(Rs > 1):
            Rs = 1
          if(Gs > 1):
            Gs = 1
          if(Bs > 1):
            Bs = 1
		  # To Nonlinear sRGB by gamma correction and scaling to 0-255 range
          r_scaled  =  (gammaCorrection(Rs)*255) + 0.5
          g_scaled =  (gammaCorrection(Gs)*255) + 0.5
          b_scaled =  (gammaCorrection(Bs)*255) + 0.5

		  # storing  new BGR values to outputImage
          outputImage[i,j] = [b_scaled, g_scaled, r_scaled]


cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()