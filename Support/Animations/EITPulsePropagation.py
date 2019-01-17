import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from cycler import cycler
import palettable
import matplotlib.animation as animation

def chi(omega, freq):
    """
    Susceptibility normalized to optical depth.

    Parameters:

    - omega: Rabi frequency
    - freq: probe detuning
    """
    # Global parameters
    # g31: excited state decoherence rate
    # g21: ground state decoherence rate
    # OD: Optical depth
    global g31, g21, OD
    Delta = 2 * np.pi * freq
    delta = Delta

    denom = abs(omega**2 + (g31 + 2j*Delta)*(g21 + 2j*delta))**2
    real = (4 * delta * (omega**2 - 4*delta*Delta) - 4*Delta * g21**2) / denom
    imag = (8 * delta**2 * g31 + 2 * g21 * (omega**2 + g21 * g31)) / denom
    return (real + 1j*imag) * g31/2 * OD/2

def H(omega, freq):
    """ Transfer function derived from chi. """
    return np.exp(1j * chi(omega, freq))

def I_out(E_in_fft, omega, f, od):
    """
    Apply transfer function in frequency domain, convert back
    to time domain and calculate output intensity.
    """
    global OD
    OD = od
    Eout_fft = E_in_fft * H(omega, f)
    Eout = np.fft.ifft(Eout_fft)
    return np.abs(Eout)**2

def update_lines(omega):
    global line_I1, line_I2, line_T1, line_T2, E_in_fft, f, OD1, OD2, \
           line_P1, line_P2
    I1 = I_out(E_in_fft, omega, f, OD1)
    line_I1.set_ydata(I1)
    line_T1.set_ydata(np.exp(-2*chi(omega, f).imag))
    line_P1.set_xdata(np.r_[line_P1.get_xdata(), -(t[I1 == I1.max()])])
    line_P1.set_ydata(np.r_[line_P1.get_ydata(), I1.max()])

    I2 = I_out(E_in_fft, omega, f, OD2)
    line_I2.set_ydata(I2)
    line_T2.set_ydata(np.exp(-2*chi(omega, f).imag))
    line_P2.set_xdata(np.r_[line_P2.get_xdata(), -(t[I2 == I2.max()])])
    line_P2.set_ydata(np.r_[line_P2.get_ydata(), I2.max()])

    txt.set_text("$\Omega_c = %.1f \gamma_2$" % (omega/(2*np.pi)))
    return line_I1, line_I2, line_T1, line_T2, line_P1, line_P2, txt

if __name__ == '__main__':
    # Set some default plotting options
    color_cycle = palettable.colorbrewer.qualitative.Set2_5.mpl_colors
    plt.matplotlib.rcParams["axes.prop_cycle"] = cycler('color', color_cycle)
    plt.matplotlib.rcParams["axes.grid"] = True
    plt.matplotlib.rcParams["lines.linewidth"] = 2
    plt.rc('grid', c='0.7', ls=':', lw=1)

    # Set simulation parameters
    OD1 = 20
    OD2 = 80
    g31 = 2 * np.pi
    g21 = 0
    omega0 = 2 * np.pi * 10
    omega0s = np.r_[np.sqrt(omega0):np.sqrt(2*np.pi*1.0):501j]**2

    # Input pulse
    t = np.r_[-30:30:10000j]
    sigma = 1 / 2.3548
    I_in = np.exp(-t**2/(2 * sigma**2))
    E_in = np.sqrt(I_in)

    # Prepare Fourier analysis
    f = np.fft.fftfreq(len(t), t[1] - t[0])
    E_in_fft = np.fft.fft(E_in)
    I_in_fft = np.abs(E_in_fft)**2

    # Prepare figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.set_xlabel("Time $(1/\gamma_2)$")
    ax.set_ylabel("Intensity (arb. u.)")
    ax.set_ylim(0, 1)
    ax.set_xlim(-5, 10)
    
    OD = OD1
    ax.plot(-t, I_in, color="gray", label="Input")
    I1 = I_out(E_in_fft, omega0, f, OD1)
    I2 = I_out(E_in_fft, omega0, f, OD2)
    line_I1, = ax.plot(-t, I1, label="OD 20")
    line_I2, = ax.plot(-t, I2, label="OD 80")
    line_P1, = ax.plot([0, -(t[I1 == I1.max()])], [1, I1.max()],
                       "--", color=ax.lines[1].get_color())
    line_P2, = ax.plot([0, -(t[I2 == I2.max()])], [1, I2.max()],
                       "--", color=ax.lines[2].get_color())
    ax.legend(loc="upper left")

    inset = inset_axes(ax, width="30%", height="30%", borderpad=1)
    inset.set_xlabel("Detuning $\Delta_p/\gamma_2$")
    inset.set_ylabel("Transmission")
    inset.text(-9, 0.1, "OD = %d" % OD)
    inset.set_xlim(-10, 10)
    OD = OD1
    line_T1, = inset.plot(f, np.exp(-2*chi(omega0, f).imag))
    OD = OD2
    line_T2, = inset.plot(f, np.exp(-2*chi(omega0, f).imag))
    txt = inset.text(-8, 0.9, "$\Omega_c = %.1f \gamma_2$" % (omega0/(2*np.pi)),
                     size=14)
    inset.plot(f, I_in_fft / I_in_fft.max(), color="gray")

    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=1800)
    ani = animation.FuncAnimation(fig, update_lines,
                                  omega0s, blit=True)
    #plt.show()
    ani.save('EITPulsePropagation.mp4', writer=writer)
