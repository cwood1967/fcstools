'''
Created on Mar 4, 2011

@author: cjw
'''

import numpy as np
import math
import time

class rawfcs(object):
    '''Work on raw fcs files. Read, trajectory, autocorrelate'''

    fclock = 20.e6

    def __init__(self, thisfilename):
        '''Create the class with a filename (or url)'''
        self.filename = thisfilename

    def readrawfile(self):

        if self.filename[0:4] == "http":
            pass
            # try:
            #     tr = time.time()
            #     handle = urllib2.urlopen(self.filename)
            #     buf = handle.read()
            #     handle.close()
            #     raw = np.frombuffer(buf, dtype=np.int32)
            #     tr2 = time.time()
            #     print("Read time :", tr2-tr)
            # except IOError:
            #     print('cannot open ', self.filename)
            #     raw = ''

        else:
            raw = np.fromfile(self.filename, dtype=np.int32)
    
        print(type(raw), len(raw))
        self.raw = raw[32:]
        self.arrivaltimes = np.cumsum(self.raw)/self.fclock

        self.lasttime= self.arrivaltimes[-1]

    def binraw(self, n, start, stop):
        self.traj, tx = np.histogram(self.arrivaltimes, bins = math.pow(2,n), range = [start,stop])
        txdel = tx[1] - tx[0]
        self.tx = tx[:-1] + txdel
        self.padfactor = 1.0
        return self.traj

    def binpad(self, bintime):
        start = self.arrivaltimes[0]
        start = 50e-9
        stop = self.arrivaltimes[-1]
        nbins = int((stop-start)/bintime) + 1
        bintraj, tx = np.histogram(self.arrivaltimes, bins =nbins, range = [start, stop + start])

        ntraj = len(bintraj)
        n2 = int(math.log(ntraj)/math.log(2))+ 1
        need = 2**n2
        imean = np.mean(bintraj)
        padlength = need - ntraj
        pad = np.zeros(padlength) + imean
        traj = np.append(bintraj, pad)
        npad = len(traj)
        print(n2)
        print(tx[0], tx[-1], tx[-2])
        delt = tx[1] - tx[0]
        thisstop = tx[0] + delt*npad
        self.padfactor = npad/ntraj
        thistx = np.arange(tx[0],thisstop,tx[1]-tx[0])
        self.tx = thistx ##tx[:-1] + (tx[1] - tx[0])/2.
        self.traj =  traj

        return traj

    def binrawdisplay(self, n, start, stop):
        self.trajdisplay, tx = np.histogram(self.arrivaltimes, bins = 2**n, range = [start,stop])
        txdel = tx[1] - tx[0]
        self.txdisplay = tx[:-1] + txdel
        return self.trajdisplay

    def autocorrelate(self, binres):
        print("start c")
        tmr0 = time.time()
        dt = self.traj - self.traj.mean()
        f1 = np.fft.fft(dt)

        fac = f1*np.conj(f1)
        acc = np.fft.ifft(fac)/self.traj.size
        acc = acc[1:len(acc)//2]
        imean = np.mean(self.traj)
        acc = 1. + self.padfactor*np.real(acc)/imean/imean
        tmr1 = time.time()
        npoints = len(acc)

        if binres == 0:
            self.actime = self.tx[1:self.tx.size/2]
            self.autocorr = acc
            print("lens ", acc.size, self.actime.size)
            return self.autocorr
        else:
            ''' bins is the number of points from the correlation to average'''
            bins = self.createLogBins(npoints, 400)

            yval = np.zeros(acc.size)
            print(len(yval), len(acc))

            acclist = []
            actimelist = []
            txindex = 0
            print("start bin")

            tmd0 = time.time()
            #d = np.digitize(self.tx[1:acc.size/2], bins)
            tmd1 = time.time()
            print(tmd1 -tmd0, " digitize")

            tt0 = time.time()
            tmr3 = time.time()
            ''' do this the way originally in idl '''
            acclist = []
            acctimelist = []
            start = 0
            stop = 0
            count = 0
            
            vsize = acc.size
            
            for bin in bins:
                stop = start + bin
                if (start >= vsize): break
                if (stop > vsize):break
                    #stop = vsize

                width = np.float(stop - start)

                temp = np.sum(acc[start:stop])/width
                #print start, stop, bin, width, temp,np.mean(self.tx[start:stop + 1])
                acclist.append(temp)
                acctimelist.append(np.mean(self.tx[start:stop + 1]))

                start = stop
                
            tmr2 = time.time()
            print("endbin")


            self.autocorr = np.asarray(acclist)
            self.actime = np.asarray(acctimelist)
            print("endc")
            print(tmr2-tmr0, "everything")
            print(tmr2-tmr3, "binloggin")
            print(tmr1 -tmr0, "correlation")
            return self.autocorr

    def createLogBins(self,ntimepoints,nbins):
        mintime = self.tx[0]
        maxtime = self.tx[ntimepoints-1]

        ''' this really needs to have correlation points to always be the same
            so do something else here -
        '''
        logmin = np.floor(math.log10(mintime))
        logmax = np.ceil(math.log10(maxtime))
        binmin = 0
        binmax = np.int(np.ceil(logmax - logmin))
        #print binmin, binmax

        '''
        bins = []
        for bin in range(binmin, binmax):
            x = np.logspace(bin,bin+1, 10,endpoint=False).astype('int')
            bins.extend(x)

        '''
        bins = np.logspace(binmin, binmax, nbins).astype('int')
        bins = np.asarray(bins)
        return bins

