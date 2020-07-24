import numpy
import csv
import FcsFit

from matplotlib import pyplot

class acCurve:
    def __init__(self, green, red):
        self.green = green
        self.red = red
        self.tau = []
        self.corr = []

def readjjlfile(filename):
    f = open(filename, 'rb')
    fcsv = csv.reader(f, delimiter=',')

    firstline = fcsv.next()
    secondline = fcsv.next()
    ncols = len(secondline)

    if secondline[-1] == "":
        ncols = ncols -1

    counter = 0

    curves = []

    while counter < len(secondline):
        if secondline[counter] != "":
            thiscurve = acCurve(firstline[counter], firstline[counter + 1])
            thiscurve.counter = counter
            curves.append(thiscurve)
            counter = counter + 3
        else:
            counter = counter +1


    ncurves = len(curves)
    curve_index = 0
    row_index = 0
    curverange = range(ncurves)
    for row in fcsv:
        #print row[0:6]
        for i in curverange:
            counter = curves[i].counter
            curves[i].tau.append(row[counter])
            curves[i].corr.append(float(row[counter + 1]))
            curves[i].channel = row[counter+2]

    f.close()
    px = 3
    py = (ncurves / 3 / 3)
    if py == 0:
        py = 1
    if py % 3 > 0:
        py = py +1
    pi = 0
    print px,py, ncurves
    pyplot.plot([0],[0])
    xtitle = ""
    colors = {1:"red", 2:"green", 21:"blue"}
    for c in curves:
        title = ":".join([c.green,c.red])
        print title
        x = numpy.asarray(c.tau, numpy.float64)
        y = numpy.asarray(c.corr, numpy.float64)

        if max(y) < 1.:
            y = y + 1.0

        indices = (x > 1.).flat
        xf1 = x.compress(indices)
        yf1 = y.compress(indices)

        indices = (xf1 < 1500.).flat
        xf = xf1.compress(indices)
        yf = yf1.compress(indices)

        fit = FcsFit.fcsfit(xf,yf)
        fit.setBounds([[0,2],[.1,1000.],[.98,1.02]]) #, [.000002,6.],[.9,1.1]])
        params = fit.fit_slsqp('1comp')

        if title <> xtitle:
            pi = pi + 1
            pyplot.subplot(py,px,pi)
            pyplot.cla()
            xtitle = title
            pyplot.title(":".join([c.green,c.red]))


        pyplot.semilogx(x, y, color=colors[int(c.channel)])

        c.g0 = params[0]
        c.td = params[1]
        c.offset = params[2]
        #c.fk = params[2]
        #c.tb = params[3]
        c.fk = 0.
        c.tb = 1.e-6

        trip = 1. + (c.fk/(1.-c.fk))*numpy.exp(-x/c.tb)
        g1 = c.g0/(1. + x/c.td)/numpy.sqrt(1. + x/(25.*c.td))
        func = g1*trip + c.offset
#        func = c.g0/(1. + x/c.td)/numpy.sqrt(1 + x/(25.*c.td)) + c.offset
        c.yfit = func
        pyplot.semilogx(x,func)

        print c.green, "," ,c.red, ",", c.channel, ",", c.g0, c.td, c.fk, c.tb, c.offset





