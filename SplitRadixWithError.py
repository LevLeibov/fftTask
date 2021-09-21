import numpy as np
import matplotlib.pyplot as plt
from SplitRadix import splitRadixFFT

'''
* In this file split radix fft is evaluated with error caused by Gaussian distribution. 
  To get the degree of error, the NMSE is calculated between the results of original and error split radix.
  NMSE is averaged by 50 realizations.
  
* For simplicity layer (stage) is defined corresponding to value N - number of points of fft in current recursive pass of algorithm.
  LAyers are defined with a dictionary called layerMap:
  {N = 64: layer = 1, N = 32: layer = 2, N = 16: layer = 3, N = 8: layer = 4, N = 4: layer = 5}
  
* Run the code to generate resulting graph NMSE(layer), which shows the layer sensitivity.
'''

# Using Gaussian distribution to add error to multiplication of input and twiddle factor
def multiplyWithError(x, y, stdCoef):
    origResult = x * y
    realPart = np.real(origResult)
    imPart = np.imag(origResult)
    sigmaReal = stdCoef * np.abs(realPart)
    sigmaIm = stdCoef * np.abs(imPart)
    realWithError = np.random.normal(realPart, sigmaReal)
    imWithError = np.random.normal(imPart, sigmaIm)
    if realPart < 0:
        realWithError *= -1
    if imPart < 0:
        imWithError *= -1
    # print("orig", realPart, imPart)
    # print(realWithError, imWithError)
    return complex(realWithError, imWithError)

def createLayers(N):
    global layerMap
    #layerMap = {64: 1, 32: 2, 16: 3, 8: 4, 4: 5}
    layerMap = {}
    power = int(np.log2(N))
    for layer in range(1, power):
        layerMap[N] = layer
        N //= 2


def splitRadixWithError(signal, N, errorLayer, stdCoef):
    if N == 1:
        return [complex(signal[0], 0)]
    if N % 2 != 0:
        print("ERROR: Length of input is not a power of 2")
        return
    if N == 2:
        return [complex(signal[0] + signal[1], 0), complex(signal[0] - signal[1], 0)]
    even = signal[::2]
    odd1 = signal[1::4]
    odd2 = signal[3::4]
    N2 = N // 2
    N3 = 3 * N // 4
    N4 = N // 4
    e = splitRadixWithError(even, N2, errorLayer, stdCoef)
    o1 = splitRadixWithError(odd1, N4, errorLayer, stdCoef)
    o2 = splitRadixWithError(odd2, N4, errorLayer, stdCoef)
    signal = np.concatenate([e, o1, o2])
    for k in range(N4):
        k_plus_N2 = k + N2
        k_plus_3N4 = k + N3
        k_plus_N4 = k + N4
        layer = layerMap[N]
        Uk = signal[k]
        Uk2 = signal[k_plus_N4]
        Zk = signal[k_plus_N2]
        Zdk = signal[k_plus_3N4]

        W1 = np.exp(-2j * np.pi / N * k)
        W3 = np.exp(-2j * 3 * np.pi / N * k)

        # create error in multiplication
        if errorLayer == layer:
            ZkW1 = multiplyWithError(W1, Zk, stdCoef)
            ZdkW3 = multiplyWithError(W3, Zdk, stdCoef)
        else:
            ZkW1 = W1 * Zk
            ZdkW3 = W3 * Zdk

        signal[k] = Uk + (ZkW1 + ZdkW3)
        signal[k_plus_N2] = Uk - (ZkW1 + ZdkW3)
        signal[k_plus_N4] = Uk2 - 1j * (ZkW1 - ZdkW3)
        signal[k_plus_3N4] = Uk2 + 1j * (ZkW1 - ZdkW3)
    return signal


def calculateError(signal, errorSignal):
    diff = 0
    sumOrig = 0
    for i in range(len(signal)):
        d = signal[i] - errorSignal[i]
        diff += d * np.conjugate(d)
        sumOrig += signal[i] * np.conjugate(signal[i])
    nmse = np.real(diff / sumOrig)
    return nmse


# Generate input signal
def generateInputSignal():
    t = range(32)
    signal = []
    for i in range(len(t)):
        signal.append(np.exp(-1j * t[i] ** 2))
    return signal


def makeErrorAveraging(signal, n, errLayer, stdCoef):
    errors = []
    numRepeats = 50
    for i in range(numRepeats):
        originSplitRadix = splitRadixFFT(signal, n)
        errorSplitRadix = splitRadixWithError(signal, n, errLayer, stdCoef)
        errors.append(calculateError(originSplitRadix, errorSplitRadix))
    return np.average(errors)


# Compare fft results for errors in different layers
def compareWithDiffLayers(stdCoef):
    signal = generateInputSignal()
    n = len(signal)
    createLayers(n)
    nmse_vals = []
    layers = []
    err = 0
    for errLayer in range(1, 6):
        err = makeErrorAveraging(signal, n, errLayer, stdCoef)
        nmse_vals.append(err)
        layers.append(errLayer)
    return [layers, nmse_vals]


# Generate input signal
signal = generateInputSignal()
n = len(signal)
signalAvg = np.abs(np.average(signal))

# Generate ffts
createLayers(n)
stdCoef = 1 * signalAvg
errorLayer = 2
originSplitRadix = splitRadixFFT(signal, n)
errorSplitRadix = splitRadixWithError(signal, n, errorLayer, stdCoef)

# Evaluate NMSE - normalized mean square error
nmse = calculateError(originSplitRadix, errorSplitRadix)
print("#2 Layer sensitivity", "\n")
print("Result of split radix: \n", originSplitRadix, "\n")
print("Result of split radix with error: \n", errorSplitRadix, "\n")
print("NMSE of signal with error = ", nmse, "\n")
print("Signal avg value = ", signalAvg, "\n")
print("RESULTS: The 1st level acts like the most sensitive to noised multiplications. However the definition of levels is important in this analysis")

# Compare fft results for different values of standard deviation (sigma) used in Gaussian distribution
stdCoefs = [0.001 * signalAvg, 0.5 * signalAvg, 1 * signalAvg, 5 * signalAvg, 10 * signalAvg]
for c in stdCoefs:
    result = compareWithDiffLayers(c)
    layers = result[0]
    nmse_vals = result[1]
    plt.plot(layers, nmse_vals, '-o', linestyle='dashed', linewidth=0.5)
    plt.legend(['sigma = 0.001*signalAvg', 'sigma = 0.5*signalAvg', 'sigma = 1*signalAvg', 'sigma = 5*signalAvg', 'sigma = 10*signalAvg'])
    plt.xticks([1, 2, 3, 4, 5])
    plt.title("Dependence NSME on layer")
    plt.xlabel("Layer")
    plt.ylabel("NSME")
plt.show()

# RESULTS:
# The 1st level acts like the most sensitive to noised multiplications. However the definition of levels is important in this analysis