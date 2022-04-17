import numpy as np
from scipy.optimize import curve_fit
import pylab as plt

# Credit: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, Vasco
# y = A*sin(Bx+C)+D - figure out A,B,C,D such that we have best fitting sine fn to our data points

# make some fake data with noise
n = 1000 # number of data points
x = np.linspace(0, 4*np.pi, n) # create a time vector, t (ind var)
y = 3.0*np.sin(x+0.001) + 0.5 + np.random.randn(n) # create fake dep var with noise

# y = A*sin(Bx+C)+D
guess_freq = 1 # freq is the B param
guess_amplitude = 3*np.std(y)/(2**0.5) # amp is the A param
guess_phase = 0 # phase is C param
guess_offset = np.mean(y) # offset is D param

p0=[guess_freq, guess_amplitude,
    guess_phase, guess_offset]

# create the function we want to fit
def my_sin(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

# now do the fit
fit = curve_fit(my_sin, x, y, p0=p0)

# we'll use this to plot our first estimate. This might already be good enough for you
y_first_guess = my_sin(x, *p0)

# recreate the fitted curve using the optimized parameters
y_fit = my_sin(x, *fit[0])

plt.plot(y, '.')
plt.plot(y_fit, label='after fitting')
plt.plot(y_first_guess, label='first guess')
plt.legend()
plt.show()