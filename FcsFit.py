'''
Created on Mar 9, 2011

@author: cjw
'''

import scipy.optimize
import numpy
import scipy

def fcs_1comp(p, *args):
    ''' d is the measured lag and g, p is the parameters
    '''
    g0 = p[0]
    td = p[1]
    offset = p[2]

    t = args[0]
    g = args[1]
    #print td
    func = g0/(1. + t/td)/numpy.sqrt(1 + t/(25.*td)) + offset

    if args[2] == 0:
        resid = g - func #p[0]/(1. + t/td)/numpy.sqrt(1 + t/(25.*td)) - offset
        res = numpy.sum(resid**2)
    else:
        res =  func

    return res

def fcs_1comptriplet(p, *args):
    ''' d is the measured lag and g, p is the parameters
    '''
    g0 = p[0]
    td = p[1]
    fk = p[2]
    tb = p[3]
    offset = p[4]

    t = args[0]
    g = args[1]

    trip = 1. + (fk/(1.-fk))*numpy.exp(-t/tb)

    g1 = g0/(1. + t/td)/numpy.sqrt(1. + t/(25.*td))
    func = g1*trip + offset
    #print p
    if args[2] == 0:
        resid = g - func #p[0]/(1. + t/td)/numpy.sqrt(1 + t/(25.*td)) - offset
        res = numpy.sum(resid**2)
    else:
        res =  func

    return res

class fcsfit():

    x = None
    y = None
    initialGuess = None
    bounds = None

    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.bounds = []

    def setGuess(self,guess):
        self.initialGuess = guess

    def setBounds(self, bounds):
        self.bounds = bounds

    def fit_slsqp(self, model):
        if model == '1comptriplet':
            fit = scipy.optimize.fmin_slsqp(fcs_1comptriplet, [.1,5., .4, .5,1.],
                                            bounds = self.bounds, acc = 1e-10,
                                            args = [self.x,self.y, 0],
                                            iprint=0, full_output=False)
        elif model == "1comp":
            fit = scipy.optimize.fmin_slsqp(fcs_1comp, [.1,5.,1.],bounds = self.bounds, acc = 1e-10,
                                            args = [self.x,self.y, 0],
                                            iprint=0, full_output=False)
        else:
            print "No model specified"
            return 0
        #print fit
        return fit
