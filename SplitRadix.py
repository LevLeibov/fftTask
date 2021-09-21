import numpy as np

'''
In this file the original split radix DIT FFT is implemented
Comparing of results of split radix FFT and np.fft.fft is printed to console
input:
  signal - input signal
  N - length of input signal
returns:
  fft of signal
'''

def splitRadixFFT(signal, N):
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
    e = splitRadixFFT(even, N2)
    o1 = splitRadixFFT(odd1, N4)
    o2 = splitRadixFFT(odd2, N4)
    signal = np.concatenate([e, o1, o2])
    for k in range(N4):
        k_plus_N2 = k + N2
        k_plus_3N4 = k + N3
        k_plus_N4 = k + N4

        Uk = signal[k]
        Uk2 = signal[k_plus_N4]
        Zk = signal[k_plus_N2]
        Zdk = signal[k_plus_3N4]

        W1 = np.exp(-2j * np.pi / N * k)
        W3 = np.exp(-2j * 3 * np.pi / N * k)

        signal[k] = Uk + (W1 * Zk + W3 * Zdk)
        signal[k_plus_N2] = Uk - (W1 * Zk + W3 * Zdk)
        signal[k_plus_N4] = Uk2 - 1j * (W1 * Zk - W3 * Zdk)
        signal[k_plus_3N4] = Uk2 + 1j * (W1 * Zk - W3 * Zdk)
    return signal

# input signal
t = range(16)
signal = []
for i in range(len(t)):
    signal.append(np.exp(-1j * t[i] ** 2))
n = len(signal)

# compare split radix fft and numpy fft
spRadixFFT = splitRadixFFT(signal, n)
numpyFFT = np.fft.fft(signal, n)
diff_arr = []
for i in range(n):
    diff = spRadixFFT[i] - numpyFFT[i]
    diff_arr.append(diff)

print("#1 Compare split radix and numpy fft", "\n")
print("Input signal is: np.exp(-1j * t ** 2)")
print("Result of split radix: \n", spRadixFFT, "\n")
print("Result of numpy fft: \n", numpyFFT, "\n")
print("Average difference of between resulting fft elements: ", np.average(diff_arr), "\n")



