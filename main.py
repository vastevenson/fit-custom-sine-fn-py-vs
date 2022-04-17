import numpy as np # np gives us the sine fn and math methods
from scipy.optimize import curve_fit # method for optimizing params for best fit (recursion)
import pylab as plt # for plotting data

# Credit: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy, Vasco
# y = A*sin(Bx+C)+D - figure out A,B,C,D such that we have best fitting sine fn to our data points

# make some fake data with noise
n = 1000 # number of data points
x = np.linspace(0, 4*np.pi, n) # create a time vector, t (ind var)
# true values
A = 3
B = 1
C = 0.001
D = 0.5
y = A*np.sin(B*x+C) + D + np.random.randn(n) # create fake dep var with noise - thanks to randn

# y = A*sin(Bx+C)+D
guess_freq = 1 # freq is the B param - this is arbitrary
guess_amplitude = np.std(y) # amp is the A param, note how our first guess will be
guess_phase = 0 # phase is C param, arbitrary guess
guess_offset = np.mean(y) # offset is D param (the vertical translation of the graph)


p0=[guess_freq,
    guess_amplitude,
    guess_phase,
    guess_offset]

# define the custom function we want to fit
def sin_fn(x, freq, amplitude, phase, offset):
    return np.sin(x * freq + phase) * amplitude + offset

# now do the fit
fit = curve_fit(sin_fn, x, y, p0=p0)

# we'll use this to plot our first estimate. This might already be good enough for you
y_first_guess = sin_fn(x, *p0)

# recreate the fitted curve using the optimized parameters
y_fit = sin_fn(x, *fit[0])

plt.plot(y, '.')
plt.plot(y_fit, label='after fitting')
plt.plot(y_first_guess, label='first guess')
plt.legend()
plt.show()