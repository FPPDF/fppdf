from global_pars import *
import numpy as np
#from scipy.special import gamma
import numba as nb

# IF the only thing that is supposed to go through gamma are scalars
# scipy.special.gamma should be equivalent to math.gamma, which should be supported by numba
from math import gamma 

@nb.njit
def I(a,b,xmin):

    if b < 100.:
        out=gamma(a+1)*gamma(b+1)                                                                   
        out=out/gamma(a+b+2)   
    else: # use Stirling's formula for numerical convergence    
        out=np.sqrt(b/(a+b+1.))*np.power(b/(a+b+1.),b)*np.power(a+b+1.,-a-1.)
        out=out*gamma(a+1.)*np.exp(a+1.)
        
    return out

@nb.njit
def Iy1(a,b,xmin):

    out=I(a,b,xmin)-2.*I(a+0.5,b,xmin)

    return out

@nb.njit
def Iy2(a,b,xmin):

    out=I(a,b,xmin)-4.*I(a+0.5,b,xmin)+4.*I(a+1.,b,xmin)

    return out

@nb.njit
def Iy3(a,b,xmin):

    out=I(a,b,xmin)-6.*I(a+0.5,b,xmin)+12.*I(a+1.,b,xmin)-8.*I(a+1.5,b,xmin)

    return out

@nb.njit
def Iy4(a,b,xmin):

    out=I(a,b,xmin)-8.*I(a+0.5,b,xmin)+24.*I(a+1.,b,xmin)-32.*I(a+1.5,b,xmin)+16.*I(a+2.,b,xmin)

    return out

@nb.njit
def Iy5(a,b,xmin):

    out=I(a,b,xmin)-10.*I(a+0.5,b,xmin)+40.*I(a+1.,b,xmin)-80.*I(a+1.5,b,xmin)+80.*I(a+2.,b,xmin)-32.*I(a+2.5,b,xmin)

    return out

@nb.njit
def Iy6(a,b,xmin):

    out=I(a,b,xmin)-12.*I(a+0.5,b,xmin)+60.*I(a+1.,b,xmin)-160.*I(a+1.5,b,xmin)+240.*I(a+2.,b,xmin)-192.*I(a+2.5,b,xmin)+64.*I(a+3.,b,xmin)

    return out

@nb.njit
def Iy7(a,b,xmin):
    
    out=I(a,b,xmin)-14.*I(a+0.5,b,xmin)+84.*I(a+1.,b,xmin)-280.*I(a+1.5,b,xmin)+560.*I(a+2.,b,xmin)-672.*I(a+2.5,b,xmin)+448.*I(a+3.,b,xmin)-128.*I(a+3.5,b,xmin)

    return out

@nb.njit
def Iy8(a,b,xmin):

    out=I(a,b,xmin)-16.*I(a+0.5,b,xmin)+112.*I(a+1.,b,xmin)-448.*I(a+1.5,b,xmin)+1120.*I(a+2.,b,xmin)-1792.*I(a+2.5,b,xmin)+1792.*I(a+3.,b,xmin)-1024.*I(a+3.5,b,xmin)+256.*I(a+4.,b,xmin)

    return out

@nb.njit
def Ic1(a,b,xmin):

    out=Iy1(a,b,xmin)

    return out

@nb.njit
def Ic2(a,b,xmin):

    out=2.*Iy2(a,b,xmin)-I(a,b,xmin)

    return out

@nb.njit
def Ic3(a,b,xmin):

    out=4.*Iy3(a,b,xmin)-3.*Iy1(a,b,xmin)

    return out

@nb.njit
def Ic4(a,b,xmin):

    out=8.*Iy4(a,b,xmin)-8.*Iy2(a,b,xmin)+I(a,b,xmin)

    return out

@nb.njit
def Ic5(a,b,xmin):

    out=16.*Iy5(a,b,xmin)-20.*Iy3(a,b,xmin)+5.*Iy1(a,b,xmin)

    return out

@nb.njit
def Ic6(a,b,xmin):

    out=32.*Iy6(a,b,xmin)-48.*Iy4(a,b,xmin)+18.*Iy2(a,b,xmin)-I(a,b,xmin)

    return out

@nb.njit
def Ic7(a,b,xmin):

    out=64.*Iy7(a,b,xmin)-112.*Iy5(a,b,xmin)+56.*Iy3(a,b,xmin)-7.*Iy1(a,b,xmin)

    return out

@nb.njit
def Ic8(a,b,xmin):

    out=128.*Iy8(a,b,xmin)-256.*Iy6(a,b,xmin)+160.*Iy4(a,b,xmin)-32.*Iy2(a,b,xmin)+I(a,b,xmin)

    return out
