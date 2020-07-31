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
                               method='trf', maxfev=5000)
        
        self.popt = popt
        self.pcov = pcov
        return popt, np.sqrt(np.diag(pcov))
    
    def predict(self, x, p=None):
        if p is not None:
            return self.__func__(x, *p)
        else:   
            return self.__func__(x, *self.popt)
    
    def compare(self, x, pcomp):
        return self.__func__(x, pcomp)
    
class fcs_2comp(fcs_1comp):
    def __init__(self, p0, bounds):
        super().__init__(p0, bounds)
        
    
    def __func__(self, t, g0, frac, td1, td2, offset):
        a1 = frac/(1. + t/td1)/np.sqrt(1 + t/(25.*td1))
        a2 = (1 - frac)/(1. + t/td2)/np.sqrt(1 + t/(25.*td2))
        func = g0*(a1 + a2) + offset
        return func

class fcs_1comp_triplet(fcs_1comp):
    def __init__(self, p0=None, bounds=None):
        super().__init__(p0, bounds)
    
    def fix(self, ftrip, ttrip):
        self.__xfunc__ = self.__func__
        self.__func__ = lambda t, g0, td, offset:self.__xfunc__(
            t, g0, td, ftrip, ttrip, offset
            )
        
    def __func__(self, t, g0, td, ftrip, ttrip, offset):
        a1 = g0/(1. + t/td)/np.sqrt(1 + t/(25.*td))
        # trip = 1 - ftrip + ftrip*np.exp(-t/ttrip)
        trip = 1 + (ftrip/(1 - ftrip))*np.exp(-t/ttrip)
        func = a1*trip + offset
        return func
    

class fcs_2comp_triplet(fcs_1comp):
    def __init__(self, p0, bounds):
        super().__init__(p0, bounds)
        
    def fix(self, ftrip, ttrip):
        self.__xfunc__ = self.__func__
        self.__func__ = lambda t, g0, frac, td1, td2, offset:self.__xfunc__(
            t, g0, frac, td1, td2, offset, ftrip, ttrip
            )
        
    def __func__(self, t, g0, frac, td1, td2, offset, ftrip, ttrip):
        a1 = frac/(1. + t/td1)/np.sqrt(1 + t/(25.*td1))
        a2 = (1 - frac)/(1. + t/td2)/np.sqrt(1 + t/(25.*td2))
        trip = 1 + (ftrip/(1 - ftrip))*np.exp(-t/ttrip)
        func = g0*(a1 + a2)*trip + offset
        return func
        
    
