'''
Created on Mar 9, 2011

@author: cjw
'''

from scipy.optimize import curve_fit
import numpy as np
import scipy

class fcs_1comp(object):
    
    def __init__(self, p0=None, bounds=None):
        self.p0 = p0
        if bounds is None:
            self.bounds = (-np.inf, np.inf)
        else:
            self.bounds = bounds

    def __func__(self, t, g0, td, offset):
        func = g0/(1. + t/td)/np.sqrt(1 + t/(25.*td)) + offset
        return func
        
    def __call__(self, x, y):
        
        popt, pcov = curve_fit(self.__func__, x, y,
                               p0=self.p0,
                               bounds=self.bounds,
                               method='trf')
        
        return popt, np.sqrt(np.diag(pcov))
    
    def predict(self, x, p):
        return self.__func__(x, *p)
    
class fcs_2comp(fcs_1comp):
    def __init__(self, p0, bounds):
        super().__init__(p0, bounds)
        
    
    def __func__(self, t, g0, frac, td1, td2, offset):
        a1 = frac/(1. + t/td1)/np.sqrt(1 + t/(25.*td1))
        a2 = (1 - frac)/(1. + t/td2)/np.sqrt(1 + t/(25.*td2))
        func = g0*(a1 + a2) + offset
        return func

class fcs_1comp_triplet(fcs_1comp):
    def __init__(self, p0, bounds):
        super().__init__(p0, bounds)
        
    def __func__(self, t, g0, td, ftrip, ttrip, offset):
        a1 = g0/(1. + t/td)/np.sqrt(1 + t/(25.*td))
        trip = 1 - ftrip + ftrip*np.exp(-t/ttrip)
        func = a1*trip + offset
        return func
        
    
