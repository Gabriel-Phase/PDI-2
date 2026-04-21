import argparse
import datetime
import sys

import cv2
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, A, x0, w, B):
    return A * np.exp(-2 * ((x - x0) ** 2) / (w ** 2)) + B


def fit_gaussian(x_vals, intensity_vals):
    vals = np.array(intensity_vals, dtype=float)
    x    = np.array(x_vals, dtype=float)

    B0   = float(np.min(vals))
    A0   = float(np.max(vals)) - B0
    x0_0 = float(x[np.argmax(vals)])
    w0   = float(x[-1] - x[0]) / 8

    try:
        popt, _ = curve_fit(_gaussian, x, vals, p0=[A0, x0_0, w0, B0], maxfev=5000)
        A, x0, w, B = popt
        span = x[-1] - x[0]
        if abs(w) < (span / 1000) or abs(w) > span:
            return {'success': False}
        return {'success': True, 'A': A, 'x0': x0, 'w': abs(w), 'B': B}
    except Exception:
        return {'success': False}


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Error: cannot open image '{path}'")
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    b = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    r = img[:, :, 2].astype(float)
    return 0.299 * r + 0.587 * g + 0.114 * b


def main():
    parser = argparse.ArgumentParser(description='Beam profile line scan from image')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--pixel-size', type=float, default=0.0356,
                        help='Physical pixel size in um/pixel (default: 0.0356)')
    args = parser.parse_args()

    gray = load_grayscale(args.image)
    h, w = gray.shape

    row = gray[h // 2]

    pixel_size_mm = args.pixel_size * 1e-3
    x_mm = (np.arange(w) - w / 2) * pixel_size_mm

    fit = fit_gaussian(x_mm, row)

    today = datetime.date.today()
    date_str = f"{today.month}/{today.day}/{today.year}"

    if fit['success']:
        diameter_um = 2 * fit['w'] * 1e3
        diam_str = f"1/e^2 Gaussian Diameter = {diameter_um:.4f} um"
    else:
        diam_str = "Fit failed"

    fig = plt.figure(figsize=(10, 5))
    plot_bottom = 0.28
    ax = fig.add_axes([0.10, plot_bottom, 0.85, 0.65])

    ax.plot(x_mm, row, color='blue', linewidth=1)

    if fit['success']:
        A  = fit['A']
        x0 = fit['x0']
        ww = fit['w']
        B  = fit['B']
        x_fine = np.linspace(x_mm[0], x_mm[-1], 1000)
        ax.plot(x_fine, _gaussian(x_fine, A, x0, ww, B), color='red', linewidth=1.5)
        level = A / np.e ** 2 + B
        ax.axhline(level, color='red', linestyle='--', linewidth=1)

    ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    ax.grid(True, alpha=0.4)
    ax.set_xlim(x_mm[0], x_mm[-1])

    sep_y = 0.265
    fig.add_artist(mlines.Line2D([0.05, 0.95], [sep_y, sep_y],
                                  transform=fig.transFigure, color='black', linewidth=0.8))

    fig.text(0.5, 0.215, 'POP beam profile', ha='center', va='top',
             fontsize=10, family='monospace')

    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.175, 0.175],
                                  transform=fig.transFigure, color='black', linewidth=0.8))

    fig.text(0.05, 0.145, date_str, ha='left', va='top', fontsize=9, family='monospace')
    fig.text(0.05, 0.10,  diam_str, ha='left', va='top', fontsize=9, family='monospace')

    plt.savefig('beam_profile.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
#  python image_scan.py C:\Gabe\Github\PDI-2\photos\image.png