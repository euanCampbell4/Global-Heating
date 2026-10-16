import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
 
SLOPED_BACKGROUND = False
SHOW_PLOT = True
 
 
def gauss(x, A, mu, sig, c):
    return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c
 
 
def gauss_slope(x, A, mu, sig, c, m):
    return gauss(x, A, mu, sig, c) + m * (x - mu)
 
 
def fit_file(path):
    # skip a header line if there is one
    try:
        d = np.loadtxt(path, delimiter=",")
    except ValueError:
        d = np.loadtxt(path, delimiter=",", skiprows=1)
    x, y = d[:, 0], d[:, 1]
 
    model = gauss_slope if SLOPED_BACKGROUND else gauss
    p0 = [y.max() - np.median(y), x[np.argmax(y)], 2.0, np.median(y)]
    if SLOPED_BACKGROUND:
        p0.append(0.0)
 
    # absolute_sigma=False: errors are scaled by the actual scatter of the data about the fit
    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
 
    centre, centre_err = popt[1], perr[1]
    fwhm, fwhm_err = 2.355 * abs(popt[2]), 2.355 * perr[2]
    resid_rms = np.std(y - model(x, *popt), ddof=len(popt))
 
    print(f"{path}")
    print(f"  centre = {centre:.3f} ± {centre_err:.3f} px")
    print(f"  FWHM   = {fwhm:.2f} ± {fwhm_err:.2f} px")
    print(f"  residual rms = {resid_rms:.2f} counts\n")
 
    if SHOW_PLOT:
        xs = np.linspace(x.min(), x.max(), 500)
        plt.figure()
        plt.plot(x, y, "k.", label="data")
        plt.plot(xs, model(xs, *popt), "r-", label=f"centre = {centre:.2f} ± {centre_err:.2f}")
        plt.xlabel("pixel")
        plt.ylabel("counts")
        plt.title(path)
        plt.legend()
    return centre, centre_err
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python gaussian_centre.py profile.csv [more files...]")
    for f in sys.argv[1:]:
        fit_file(f)
    if SHOW_PLOT:
        plt.show()