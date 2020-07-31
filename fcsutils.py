import math
import time
import os

import numpy as np
import pandas as pd

class FcsTrajectory(object):
    '''Work on raw fcs files. Read, trajectory, autocorrelate'''

    fclock = 20.e6

    def __init__(self, thisfilename, bintime):
        '''Create the class with a filename (or url)'''
        self.filename = thisfilename
        self.bintime = bintime

    def __call__(self):
        self.readrawfile()
        self.bin(self.bintime, display=False)
        self.bin(self.lasttime/1000, display=True)
        self._autocorrelate(1)
        
    def readrawfile(self):

        if self.filename[0:4] == "http":
            raw = 0
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
    
        self.raw = raw[32:]
        self.arrivaltimes = np.cumsum(self.raw)/self.fclock

        self.lasttime= self.arrivaltimes[-1]

    def binraw(self, n, start, stop):
        self.traj, tx = np.histogram(self.arrivaltimes,
                                     bins = math.pow(2,n),
                                     range = [start,stop])
        txdel = tx[1] - tx[0]
        self.tx = tx[:-1] + txdel
        self.padfactor = 1.0
        return self.traj

    def bin(self, bintime, display=False):
        start = 0
        stop = self.arrivaltimes[-1] + bintime
        bins = np.arange(start, stop, bintime)
        bintraj, tx = np.histogram(self.arrivaltimes, bins=bins)

        if display:
            self.trajdisplay = bintraj
            self.txdisplay = tx[:-1]
        else:
            self.traj = bintraj
            self.tx = tx[:-1]
            self.padfactor = 1
            
        return bintraj, tx[:-1]
    
    def padtraj(self):
        ntraj = len(self.traj)
        n2 = int(math.log(ntraj)/math.log(2))+ 1
        need = 2**n2
        imean = self.traj.mean()
        padded = np.zeros(need) + imean
        padded[:ntraj] = self.traj
        npad = len(padded)
        self.padfactor = npad/ntraj
        
        dtime = self.tx[1] - self.tx[0]
        total_time = npad*dtime
        thistx = np.linspace(self.tx[0], total_time, npad)
        self.txpadded = thistx
        self.trajpadded =  padded
        return


    def _autocorrelate(self, binres):
        # print("start c")
        tmr0 = time.time()
        self.padtraj()
        dt = self.trajpadded - self.trajpadded.mean()
        f1 = np.fft.fft(dt)

        fac = f1*np.conj(f1)
        acc = np.fft.ifft(fac)/len(self.trajpadded)
    
        acc = acc[1:len(acc)//2]
        imean = np.mean(self.trajpadded)
        acc = 0. + self.padfactor*np.real(acc)/imean/imean

        self.rawactime = self.txpadded[1:len(acc) + 1]
        self.rawautocorr = acc
        
        if binres:
            # print("start bin")
            tmr3 = time.time()
            bins = self.createLogBins(len(acc), 160)
            dzbins = np.digitize(self.rawactime, bins)
            
            udz = np.unique(dzbins)
            
            acclist = []
            acctimelist = []
            
            for u in udz:
                dzb = np.where(dzbins == u)
                time_points = self.rawactime[dzb]
                acc_points = self.rawautocorr[dzb]
                dt = time_points.mean()     

                temp = np.sum(acc_points)/len(acc_points)
                acclist.append(temp)
                acctimelist.append(dt)
                
            tmr2 = time.time()
            # print("endbin")


            self.autocorr = np.asarray(acclist)
            self.actime = np.asarray(acctimelist)
            # print("endc")
            # print(tmr2-tmr0, "everything")
            # print(tmr2-tmr3, "binloggin")
            # print(tmr1 -tmr0, "correlation")
            return self.autocorr

    def createLogBins(self,ntimepoints,nbins):
        mintime = self.rawactime[0]
        maxtime = self.rawactime[-1]
        logmin = math.log10(mintime)
        logmax = math.log10(maxtime)
        
        bins = np.logspace(logmin, logmax, nbins, endpoint=False)
        self.acbins = bins
        return self.acbins
    
    def ac_as_dataframe(self):

        return pd.DataFrame({'file':os.path.basename(self.filename),
                             'tau':self.actime,
                             'autocorr':self.autocorr})        


def readfcsfile(filename):
    
    corr_start = 'CorrelationArraySize'
    corrsize = 'CorrelationArray ='
    corr_end = 'PulseDistanceHistogramArraySize'
    print(filename)
    fcslines = open(filename, errors='replace').readlines()
    
    starts = [(i, line.strip()) for i, line in enumerate(fcslines) if corrsize in line]
    stops = [(i, line.strip()) for i, line in enumerate(fcslines) if corr_end in line]
    
    df_list = list()
    for i, num in enumerate(starts):
        first = num[0] + 1
        asize = int(fcslines[num[0]].split('=')[1].split()[0])
        last = first +  asize ##stops[i][0] - 1
        _df = lines_to_df(fcslines[first:last])
        _df['series'] = i
        _df['file'] = os.path.basename(filename)
        df_list.append(_df)
    return pd.concat(df_list) 

def lines_to_df(lines):
    
    taulist = list()
    aclist = list()
    for line in lines:
        t, ac = line.strip().split()
        taulist.append(float(t))
        aclist.append(float(ac) - 1.)
            
    df = pd.DataFrame({'tau':taulist, 'autocorr':aclist})
    return df