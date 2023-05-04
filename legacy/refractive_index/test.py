from phasetorch._refindxdata import calculate_delta_vs_E, calculate_beta_vs_E
import numpy as np
try:
    from matplotlib import pyplot as plt
    mplotlib_import = True
except:
    mplotlib_import = False
    print("no matplotlib detected. results will not be plotted.")

"""
the users will have to input the density and the chemical
formula of the element or compound. the formula could be a
string or a dictionary with keys as the elements and values
as the number of atoms. e.g. Si3N4 could be passed as
1. "Si3N4"
2. {"Si" : 3, "N": 4}
"""
density = 3.44
formula = "Si3N4"

# density = 1.2
# formula = "C16H14O3"

"""
energy range between 10 and 30 kev. the library support values
upto 400 keV for most elements, but for this example comparison,
we ony go up to 30 keV because those values are available from
an alternate source for comparison
"""
energy_vector = np.linspace(10,25,50)

beta_phasetorch  = calculate_beta_vs_E(density, formula, energy_vector)

delta_phasetorch = calculate_delta_vs_E(density, formula, energy_vector)

data_henke = np.loadtxt("henke_Si3N4.txt")
# data_henke = np.loadtxt("polycarbonate.txt")

# energies for henke data are tabulated in ev
# converting to keV
energy_henke = data_henke[:,0]*1e-3

beta_henke, delta_henke = data_henke[:,2], data_henke[:,1]

beta_henke = np.interp(energy_vector, energy_henke, beta_henke)

delta_henke = np.interp(energy_vector, energy_henke, delta_henke)

if mplotlib_import:
    fig1, ax1 = plt.subplots(2)
    ax1[0].plot(energy_vector, beta_phasetorch, '-k')
    ax1[0].scatter(energy_vector, beta_henke,\
    marker = "o", edgecolor="r", facecolors="None")
    ax1[0].set_ylabel(r"$\beta$")
    ax1[0].set_xlabel(r"Energy (keV)")
    ax1[0].legend(["PhaseTorch", "Henke"])

    ax1[1].plot(energy_vector, \
    100*(beta_phasetorch-beta_henke)/beta_henke,\
    "-k")
    ax1[1].set_ylabel("% difference")
    ax1[1].set_xlabel(r"Energy (keV)")

    plt.show(block=False)
    plt.savefig('%s_beta.png' % formula, format='png')

    fig2, ax2 = plt.subplots(2)
    ax2[0].plot(energy_vector, delta_phasetorch, '-k')
    ax2[0].scatter(energy_vector, delta_henke,\
    marker = "o", edgecolor="r", facecolors="None")
    ax2[0].set_ylabel(r"$\delta$")
    ax2[0].set_xlabel(r"Energy (keV)")
    ax2[0].legend(["PhaseTorch", "Henke"])

    ax2[1].plot(energy_vector, \
    100*(delta_phasetorch-delta_henke)/delta_henke,\
    "-k")
    ax2[1].set_ylabel("% difference")
    ax2[1].set_xlabel(r"Energy (keV)")

    plt.show(block=False)
    plt.savefig('%s_delta.png' % formula, format='png')
