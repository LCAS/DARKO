# -*- coding: utf-8 -*-
"""
Copyright University of Lincoln, UK (2020)

"""

import numpy as np
from numpy.linalg import norm
import scipy
from numpy import absolute as abs
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R

class qualitative_rep:

	def __init__(self):
		self.PLUS = 1
		self.ZERO = 0
		self.MINUS = -1
        self.dist_threshold = 0 
        self.speed_threshold = 0 
        self.tait_bryan_threshold = 0 
        self.maa_angle_threshold = 0
        self.option = 'qtc_c2'
        self.threshold= 10e-08


    def getSymbol(self, Dnow, Dnext):

    	if (abs(Dnow - Dnext) < self.threshold):
    		symbol = self.ZERO
    	elif (Dnow > Dnext):
    		symbol = self.MINUS
    	else symbol = self.PLUS 

        return symbol 


    def quantizeAndGet(self, theta):

        if (abs(theta) < self.threshold):
            s = self.ZERO
        elif (theta > 0):
            s = self.PLUS
        else:
            s = self.MINUS

        return s


    def getAngle(self, a, b):
        angle = np.arctan2(norm(np.cross(a,b)),np.dot(a,b))
        return angle



############################################################################################################################################################################
#																																										   #
#																																										   #	
############################################################################################################################################################################



    def getAB_3D(self, O1now, O1next, O2now, O2next):


       # GETAB_2F get the A and B property but use only the current and next frame
       # In this variant of getAB, we are only using the current and next
       # position of object O1 and the current position of object O2. In other
       # words, we disregard where O1 came from. A '+' symbol means that O1 has
       # moved closed to O2 during the transition t-->t+1. A '-' symbol means
       # that it moved away from O2. A '0' means that the distance is the same,
       # hinting at the fact that either O1 remained static or that it moved
       # across a circle centered at O2. Note that this works for an arbitrary
       # number of dimensions, but we have named it 3D to make sure that we are
       # referring to the bird data case
     

    	Dnow = pdist([O1now, O2now]) # pdist(X, metric='euclidean') --> X is an m by n array of m original observations in an n-dimensional space.
    	Dnext1 = pdist([O1next, O2now])
    	Dnext2 = pdist([O2next, O1now])

    	s1 = self.getSymbol(Dnow, Dnext1, threshold)
    	s2 = self.getSymbol(Dnow, Dnext2, threshold)


        return [s1, s2]  


    def getC_3D(self, Know, Knext, Lnow, Lnext ):
    # GETC_2F Get the C property of objects K and L
    #    This is identical to the 2D case. Note that we are using the positions
    #    of the objects, rather than their 'instance' speeds (ds / dt) because
    #    we are operating on the discrete domain anyway. 

        dK = pdist([Know, Knext])
        dL = pdist([Lnow, Lnext])
        
        if (abs(dL - dK) < threshold):
            symbol = self.ZERO
        elif (dK > dL):
            symbol = self.PLUS
        else:
            symbol = self.MINUS

        return symbol



    def getF_3D(self, K, Knext, L, Lnext ):
    # GETF_3D Get the angle property. 
    #    This is again identical to the 2D case
            
        angle1 = self.getAngle((Knext - K), (L - K))
        angle2 = self.getAngle((Lnext - L), (K - L))
        
        if (abs(angle2 - angle1) < self.threshold):
            symbol = self.ZERO;
        elif (angle2 > angle1):
            symbol = self.PLUS
        else:
            symbol = self.MINUS




    def getGHI_3D(self, P1prev, P1now, P1next, P2prev, P2now, P2next):
        # GETDFE This is the alternative to the side constraints, as devised for 3D
        #    Since it is not possible to define a side of a line in 3D, we are
        #    instead opting to use the Tait-Bryan angles of the transformation of
        #    the two Frenet-Serret frames. The symbols +, -, and 0, are defined by
        #    an appropriate quantization (tolerance).  


        # t1(T)
        v = P1next - P1now
        vn = norm(v) 
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0           
        t1 = v / vn
        
        # t2(T)
        v = P2next - P2now
        vn = norm(v)
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0
        t2 = v / vn
        
        # t1(T-1)
        v = P1now - P1prev
        vn = norm(v)
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0

        t1prev = v / vn
        
        # t2(T-2)
        v = P2now - P2prev
        vn = norm(v)
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0
        t2prev = v / vn
        
        # b1(T)
        v = np.cross(t1prev,t1)
        vn = norm(v)
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0
        b1 = v / vn
        
        # b2(T)
        v = np.cross(t2prev,t2)
        vn = norm(v)
        if (vn < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0    
        b2 = v / vn
        
        # n1(T)
        n1 = np.cross(b1,t1)
        
        # n2(T)
        n2 = np.cross(b2,t2)

        # Discrete Frenet-Serret 
        F1 = [t1 n1 b1]
        F2 = [t2 n2 b2]
        
        # Transformation matrix (get one frame aligned with the other)
        d = np.det(F1)
        if (abs(d) < self.tolerance):
            s1 = 0 
            s2 = 0 
            s3 = 0
            
        T = F2*(F1**(-1))
        
        # Tait-Bryan
        alpha = np.arctan2d(T(2,1),T(1,1))
        beta = np.arctan2d(-T(3,1), np.sqrt(T(3,2)**2 + T(3,3)**2))
        gamma = np.arctan2d(T(3,2), T(3,3))
        
        # symbols
        s1 = self.quantizeAndGet(alpha)
        s2 = self.quantizeAndGet(beta)
        s3 = self.quantizeAndGet(gamma)

        return [s1, s2, s3]




    def getQTC_3D(self, T1, T2):
        # GETQTC_3D Similar to the 2D case, but for 3D
        #    Major differences to the 2D case: a) 3D instead of 2D (duh!), b) the
        #    distance constraint does not utilize the previous frame, and c) the
        #    side constraints have been substituted by their 3D counterpart

        if (T1.shape[1] != 3 or T2.shape[1] != 3 or T1.shape[0] != T2.shape[0] or T1.shape[0] < 3):  # time series
            print('T1 and T2 must each be nx3 matrices, with the X coordinates ')
            print('in the first column and the Y coordinates in the second ')
            print('column.\nAlso there must be at least 3 rows\n')
            print('Returning empty ...\n')
            QTC = np.array([])


        if (self.option.find('qtc_b1') == 0 and self.option.find('qtc_b2') == 0 and self.option.find('qtc_c1') == 0 and self.option.find('qtc_c2')==0):    
            print('Invalid option. Choose among the following: \n')
            print('\t1. QTC_B1 --> [A B]\n')
            print('\t2. QTC_B2 --> [A B C]\n')
            print('\t3. QTC_C1 --> [A B G H I]\n')
            print('\t4. QTC_C2 --> [A B C G H I F]\n')
            print('Returning empty ...\n')
            QTC = np.array([])


        len = T1.shape[0]

        if self.option == 'qtc_b1':
            QTC = np.zeros(len,2);
            for i in range(2:(len-1)):
                QTC[i,:] = self.getAB_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.dist_threshold)            
        elif self.option =='qtc_b2':
            QTC = np.zeros(len,3);
            for i in range(2:(len-1)):
                [A, B] = self.getAB_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.dist_threshold)                        
                C = self.getC_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.speed_threshold)            
                QTC[i,:] = [A B C]
        elif self.option =='qtc_c1':
            QTC = np.zeros(len,5)
            for i in range(2:(len-1)):
                [A, B] = self.getAB_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.dist_threshold)    
                [G, H, I] = self.getGHI_3D(T1(i-1,:), T1(i,:), T1(i+1,:), T2(i-1,:), T2(i,:), T2(i+1,:), self.tait_bryan_threshold)
                QTC[i,:] = [A B G H I]

        elif self.option == 'qtc_c2':
            QTC = np.zeros(len,7)
            for i in range(2:(len-1)): 
                [A, B] = self.getAB_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.dist_threshold)    
                C = self.getC_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.speed_threshold)  
                [G, H, I] = self.getGHI_3D(T1(i-1,:), T1(i,:), T1(i+1,:), T2(i-1,:), T2(i,:), T2(i+1,:), self.tait_bryan_threshold)
                F = self.getF_3D(T1(i,:), T1(i+1,:), T2(i,:), T2(i+1,:), self.maa_angle_threshold)
                QTC[i,:] = [A B C G H I F]
        else:
            print('NO option for QTC type has been defined!')
            
        return QTC




    def QTC_To_Index(self, QTC):
        # QTC_TO_INDEX Transform QTC tuple to an index
        #    Map each symbol to a digit in [0..2], i.e. use a tertiary base. Convert
        #    to decimal. Finally add 1 so that the result corresponds to a MATLAB
        #    index. For example, QTC = [+ 0 0 - 0 + +] --> [2 0 0 1 0 2 2] --> 1493
        #    --> 1494. Any QTC_C2 in 3D has 7 symbols, so the result will be between
        #    1 and 3^7
        summ = 0
        for i in range(np.size(QTC):-1:1):
            power = np.size(QTC) - i
            switch(QTC(i))
            if  QTC(i) == self.ZERO:
                base = 0
            elif QTC(i) == self.MINUS:
                base = 1
            elif QTC(i) == self.PLUS:
                base = 2
            else: 
                print('unknown sign for QTC relations')
            summ = summ + base*(3**power)
        
        summ = summ + 1
       return summ




if __name__ == "__main__":

    qltv_rep = qualitative_rep()
    T1 = np.array([]) # time series traj of moving object k
    T2 = np.array([]) # time series traj of moving object l


    QTC = qltv_rep.getQTC_3D(T1, T2)
    summ = qltv_rep.QTC_To_Index(QTC)
