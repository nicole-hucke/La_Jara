import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

""" 
This code removes spikes from the Accoustic Doppler Velocimeter (ADV) measurements. 
It is a python adaptation of the code from Rashedul Islam (2024). 
It uses bivariate kernel density function to separate the data cluster from the spike clusters. 
The elements identified as spikes are removed and replaced by the linearly interpolated values.

References:
Rashedul Islam (2024). Despiking Acoustic Doppler Velocimeter (ADV) Data 
(https://www.mathworks.com/matlabcentral/fileexchange/39767-despiking-acoustic-doppler-velocimeter-adv-data), 
MATLAB Central File Exchange. Retrieved February 26, 2024.

"""

def despike_adv(u, hx=0.01, hy=0.01):
    N = len(u)

    du = np.zeros(N)
    for i in range(1, N-1):
        db = u[i] - u[i-1]
        df = u[i+1] - u[i]
        if abs(db) > abs(df):
            du[i] = df
        else:
            du[i] = db

    u1 = np.copy(u)
    w1 = np.copy(du)
    # add the last data point to u1 and w1
    u1[-1] = u[-1]
    w1[-1] = 0

    print(u1.shape)
    print(w1.shape)

    # axis rotation
    th = np.arctan2((N * np.sum(u1 * w1) - np.sum(u1) * np.sum(w1)),
                    (N * np.sum(u1 * u1) - np.sum(u1) * np.sum(u1)))
    ut = (u1) * np.cos(th) + (w1) * np.sin(th)
    wt = -(u1) * np.sin(th) + (w1) * np.cos(th)
    data = np.column_stack((ut, wt))

    # applying kernel density function using Botev et al.'s algorithm
    d, X, Y = kde2d_botev(data, N, hx, hy)

    uf, wf = X[0, :], Y[:, :]
    dp = np.max(np.max(d))
    wp, up = np.unravel_index(np.argmax(d), d.shape)

    # calculating cut-off threshold
    c1 = 0.4
    c2 = 0.4
    ul, uu = cutoff1(dp, uf, c1, c2, d[wp, :], up)
    wl, wu = cutoff1(dp, wf, c1, c2, d[:, up], wp)

    # calculating axes of ellipse and identifying spikes
    uu1 = uu - 0.5 * (uu + ul)
    ul1 = ul - 0.5 * (uu + ul)
    wu1 = wu - 0.5 * (wu + wl)
    wl1 = wl - 0.5 * (wu + wl)
    Ut1 = ut - 0.5 * (uu + ul)
    Wt1 = wt - 0.5 * (wu + wl)
    F = np.zeros(N)

    at = 0.5 * (uu1 - ul1)
    bt = 0.5 * (wu1 - wl1)
    for i in range(N):
        if Ut1[i] > uu1 or Ut1[i] < ul1:
            F[i] = 1
        else:
            we = np.sqrt((bt**2) * (1 - (Ut1[i]**2) / (at**2)))
            if Wt1[i] > we or Wt1[i] < -we:
                F[i] = 1

    Id = np.where(F > 0)[0]

    # replacing spikes by linearly interpolated values
    if Id[0] == 0:
        Id = Id[1:]
        u[0] = np.mean(u)
    if Id[-1] == N-1:
        Id = Id[:-1]
        u[N-1] = np.mean(u)

    u1 = vel_replace(u, Id)

    return u1, Id

def cutoff1(dp, uf, c1, c2, f, Ip):
    lf = len(f)
    dk = np.append(0, np.diff(f)) * 256 / dp

    for i in range(Ip-1, 1, -1):
        if f[i] / f[Ip] <= c1 and abs(dk[i]) <= c2:
            i1 = i
            break

    for i in range(Ip+1, lf-1):
        if f[i] / f[Ip] <= c1 and abs(dk[i]) <= c2:
            i2 = i
            break

    ul = uf[i1]
    uu = uf[i2]
    return ul, uu

def vel_replace(u, Id):
    N = len(u)
    I1 = np.append(Id, N)
    
    for i in range(len(I1) - 1):
        for j in range(i, len(I1) - 1):
            if I1[j + 1] - I1[j] > 1:
                break
        ds = j - i + 2
        du1 = (u[I1[j] + 1] - u[I1[i] - 1]) / ds
        u[I1[i]] = u[I1[i] - 1] + du1

    return u

def kde2d_botev(data, N, hx=0.01, hy=0.01):
    global I, A2
    n = 256

    # calculating scaled data
    MAX_XY = np.max(data, axis=0)
    MIN_XY = np.min(data, axis=0)
    scaling = MAX_XY - MIN_XY
    transformed_data = (data - MIN_XY) / scaling

    # bin the data uniformly using a regular grid
    initial_data = ndhist(transformed_data, n)
    # discrete cosine transform of initial data
    a = dct2d(initial_data)
    # now compute the optimal bandwidth^2
    I = np.arange(n)**2
    A2 = a**2
    t_star = fsolve(evolve, [0.05], args=(N,))[0]
    p_02 = func([0, 2], t_star, N)
    p_20 = func([2, 0], t_star, N)
    p_11 = func([1, 1], t_star, N)
    t_y = hy**2
    t_x = hx**2

    # smooth the discrete cosine transform of initial data using t_star
    a_t = np.exp(-np.arange(n)**2 * np.pi**2 * t_x / 2)[:, None] * np.exp(
        -np.arange(n)**2 * np.pi**2 * t_y / 2
    ) * a
    # now apply the inverse discrete cosine transform
    density = idct2d(a_t) * (np.prod(scaling)) / len(a_t)

    X, Y = np.meshgrid(
        np.linspace(MIN_XY[0], MAX_XY[0], n),
        np.linspace(MIN_XY[1], MAX_XY[1], n),
    )

    return density, X, Y

def ndhist(data, M):
    nrows, ncols = data.shape
    bins = np.zeros((nrows, ncols), dtype=int)

    for i in range(ncols):
        hist, edges = np.histogram(data[:, i], bins=np.linspace(0, 1, M+1))
        bins[:, i] = np.digitize(data[:, i], bins=edges[1:])  # Use 1-based indexing

    # Combine the vectors of 1D bin counts into a grid of nD bin counts
    binned_data = np.zeros((M,) * ncols, dtype=float)
    for i in range(nrows):
        binned_data[tuple(bins[i] - 1)] += 1 / nrows  # Use 0-based indexing

    return binned_data

def evolve(t, N):
    Sum_func = func([0, 2], t, N) + func([2, 0], t, N) + 2 * func([1, 1], t, N)
    time = (2 * np.pi * N * Sum_func)**(-1/3)
    out = (t - time) / time
    return out

def func(s, t, N):
    if sum(s) <= 4:
        Sum_func = func([s[0] + 1, s[1]], t, N) + func([s[0], s[1] + 1], t, N)
        const = (1 + 1 / 2**(sum(s) + 1)) / 3
        time = (-2 * const * K(s[0]) * K(s[1]) / N / Sum_func)**(1 / (2 + sum(s)))
        out = psi(s, time)
    else:
        out = psi(s, t)
    return out

def psi(s, Time):
    global I, A2
    w = np.exp(-I * np.pi**2 * Time) * np.concatenate(([1], 0.5 * np.ones(len(I) - 1)))
    wx = w * (I**s[0])
    wy = w * (I**s[1])
    out = (-1)**sum(s) * np.dot(wy, A2.dot(wx)) * np.pi**(2 * sum(s))
    return out

def K(s):
    out = (-1)**s * np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
    return out

import numpy as np

def dct2d(data):
    # computes the 2 dimensional discrete cosine transform of data
    # data is an nd cube
    nrows, ncols = data.shape
    if nrows != ncols:
        raise ValueError('data is not a square array!')

    # Compute weights to multiply DFT coefficients
    w = np.concatenate(([1], 2 * np.exp(-1j * np.arange(1, nrows) * np.pi / (2 * nrows))))
    weight = w[:, np.newaxis] * np.ones((1, ncols))

    def dct1d(x):
        # Re-order the elements of the columns of x
        x = np.concatenate((x[::2, :], x[::-2, :]))

        # Multiply FFT by weights
        transform1d = np.real(weight * np.fft.fft(x, axis=0))
        return transform1d

    data = dct1d(dct1d(data).T).T

    return data

def idct2d(data):
    # computes the 2 dimensional inverse discrete cosine transform
    nrows, ncols = data.shape

    # Compute weights
    w = np.exp(1j * np.arange(nrows) * np.pi / (2 * nrows))
    weights = w[:, np.newaxis] * np.ones((1, ncols))

    def idct1d(x):
        y = np.real(np.fft.ifft(weights * x, axis=0))
        out = np.zeros((nrows, ncols))
        out[::2, :] = y[:nrows//2, :]
        out[1::2, :] = y[nrows-1:nrows//2:-1, :]
        return out

    data = idct1d(idct1d(data).T).T

    return data

####### function to load CSV data and execute despike algorithm ######
def load_and_despike(csv_file_path, hx=0.01, hy=0.01):
    data = pd.read_csv(csv_file_path)
    data = data.apply(pd.to_numeric, errors='coerce')

    vx_series = data['VelX'].iloc[1:].values # ignoring first two columns
    vy_series = data['VelY'].iloc[1:].values 
    vz_series = data['VelZ'].iloc[1:].values

    vx_despiked, vx_spike_indices = despike_adv(vx_series, hx, hy) # call the despike_adv function
    vy_despiked, vy_spike_indices = despike_adv(vy_series, hx, hy)
    vz_despiked, vz_spike_indices = despike_adv(vz_series, hx, hy)

    # print or plot the results as needed
    print("Vel X Spike indices:", vx_spike_indices)
    print("Vel Y Spike indices:", vy_spike_indices)
    print("Vel Z Spike indices:", vz_spike_indices)

    # plot the original and despiked series
    fig, axs = plt.subplots(3, figsize=(10, 15))
    # vx
    axs[0].plot(vx_series, label='Original Vel X')
    axs[0].plot(vx_despiked, label='Despiked Vel X')
    axs[0].scatter(vx_spike_indices, vx_despiked[vx_spike_indices], color='red', label='Spikes')
    axs[0].set_title('Original and Despiked Velocity in X')
    axs[0].legend()

    # vy
    axs[1].plot(vy_series, label='Original Vel Y')
    axs[1].plot(vy_despiked, label='Despiked Vel Y')
    axs[1].scatter(vy_spike_indices, vy_despiked[vy_spike_indices], color='red', label='Spikes')
    axs[1].set_title('Original and Despiked Velocity in Y')
    axs[1].legend()

    # vz
    axs[2].plot(vz_series, label='Original Vel Z')
    axs[2].plot(vz_despiked, label='Despiked Vel Z')
    axs[2].scatter(vz_spike_indices, vz_despiked[vz_spike_indices], color='red', label='Spikes')
    axs[2].set_title('Original and Despiked Velocity in Z')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


csv_file_path = 'T1A_SP23_ADV.csv'
load_and_despike(csv_file_path)

print(data.columns)