try:
    import cupy as cp
    import numpy as np
    cuda_on = True
except:
    import numpy as np
    cuda_on = False

class Sft:

    def __init__ (self, NB, m, inv=False, CtrBtwnPix=False):

        """
            Slow Fourier Transform, using the theory described in [1]_. 
            Assumes the original array is square. 

            Parameters
            ----------
        
            NB : int
                 the linear size of the resulting array (integer)
        
            m : float
                m/2 = maximum spatial frequency to be computed (in lam/D)
        
            inv : boolean (default=False)
                  boolean (direct or inverse) see the definition of isft()
            
            CtrBtwnPix : boolean (default=False)
                         type of centering for the disk. If True, the disk is centered between
                         four pixels.
               
            References
            ---------
        
            .. [1] Soummer, Pueyo, Sivaramakrishnan, Vanderbei, Fast computation 
                of Lyot-style coronagraph propagation, Optics Express, vol. 15, issue 24, 
                p. 15935 (2007).
                https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935
        
        """

        # Simply initialize parameters here
        self.NB = NB
        self.m = m
        self.inv = inv
        self.CtrBtwnPix = CtrBtwnPix


    def sft(self, A2):
        """
            Slow Fourier Transform, using the theory described in [1]_. 
            Assumes the original array is square. 
            Computations will be done on GPU if input array is a cupy array

            Parameters
            ----------
            A2 : array_like
                 the 2D original array
        
            Returns
            ---------
            res : array_like
                  Fourier transform of the array A2 within array of dimensions NBxNB
        """

        # First determine type (cupy or numpy) of array A2
        xp = get_array_module(A2)

        # Initialize some parameters
        val    = 0
        if self.CtrBtwnPix is True:
            val = 1/2
        NA    = np.shape(A2)[0]
        coeff = self.m/(NA*self.NB)
        
        sign = -1.0
        if self.inv:
            sign = 1.0

        U = xp.zeros((1,self.NB))
        X = xp.zeros((1,NA))
        
        X[0,:] = (1./NA)*(xp.arange(NA)-NA/2.+val)
        U[0,:] =  (self.m/self.NB)*(xp.arange(self.NB)-self.NB/2.+val)
           
        XU = 2.*np.pi* X.T.dot(U)
        A3 = xp.exp(1j*XU) # sign*1j*xp.sin(XU)  +xp.cos(XU)
        A1 = A3.T
        
        B  = A1.dot(A2.dot(A3))

        return coeff*B



def get_array_module(arr):
    '''
    if cuda is available, and arr is a cupy array, returns cupy, otherwise returns numpy
    '''
    if (cuda_on):
        xp = cp.get_array_module(arr)
    else:
        xp = np
    return(xp)

