import numpy as np

def create_block(width, height, res):
    top = np.linspace([-width/2, height/2], [width/2, height/2], int(0.5*res*width/(width+height)))[:-1,:]
    right = np.linspace([width/2, height/2], [width/2, -height/2], int(0.5*res*height/(width+height)))[:-1,:]
    bottom = np.linspace([width/2, -height/2], [-width/2, -height/2], int(0.5*res*width/(width+height)))[:-1,:]
    left = np.linspace([-width/2, -height/2], [-width/2, height/2], int(0.5*res*height/(width+height)))[:-1,:]
    return np.concatenate((top, right, bottom, left))

def create_naca_airfoil(k, length, res, alpha=0):
    a, b, c, d = tuple(int(c) for c in str(k))
    t = 10*c+d
    p = 0.1*b
    m = 0.01*a
    xa = np.linspace(0, length, res//2)
    xp = np.linspace(0, 1, res//2)
    xa = np.insert(xa, 1, xa[1]/3)
    xp = np.insert(xp, 1, xp[1]/3)
    yt = 5*t*0.01*(0.2969*np.sqrt(xp) - 0.1260*xp - 0.3516*xp**2 + 0.2843*xp**3 - 0.1036*xp**4)
    yc = np.piecewise(xp, [xp<=p, xp>p], [lambda x: m*(2*p*x-x**2)/(p**2), lambda x: m*(1-2*p+2*p*x-x**2)/((1-p)**2)])
    the = np.arctan(np.piecewise(xp, [xp<=p, xp>p], [lambda x: 2*m*(p-x)/(p**2), lambda x: 2*m*(p-x)/((1-p)**2)]))
    yu = yc + yt*np.cos(the)
    yl = yc - yt*np.cos(the)
    y = np.concatenate((yu[:-1], yl[-1::-1]))*length
    x = np.concatenate((xa[:-1], xa[-1::-1]))-(length/2)
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array(((c, s), (-s, c)))
    return np.transpose(R @ np.array([x, y]))