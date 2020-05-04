#!/usr/bin/env python3

""" Python implementation of Whitley-Stroud equations for Cascade parametric emission.


Whitley, R., Stroud, C. (1976). Double optical resonance Physical Review A  14(4), 1498-1513. https://dx.doi.org/10.1103/physreva.14.1498
"""
import numpy as np
from scipy.ndimage import convolve1d
from scipy.stats import cauchy


__author__ = "Alessandro Cerè"
__copyright__ = "Copyright (c) 2020 Alessandro Cere"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Alessandro Cerè"
__status__ = "Development"


DeltaT = 30e-9
Gamma1 = 6.066
Gamma2 = 0.666


def s_33(Oma, Omb, Gammaa, Gammab, delta, Delta):
    return (
        Oma ** 2
        * Omb ** 2
        * (
            Gammaa * Gammab * ((delta - Delta) ** 2 + (Gammaa + Gammab) ** 2)
            + Gammaa * (Gammaa + Gammab) * Oma ** 2
            + (Gammaa + Gammab) ** 2 * Omb ** 2
        )
    ) / (
        Delta ** 4 * Gammaa * Gammab ** 3
        + delta ** 4 * Gammaa * Gammab * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
        - 2
        * delta ** 3
        * Delta
        * Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2 + Omb ** 2)
        + (Gammab * (Gammaa + Gammab) + Oma ** 2 + Omb ** 2)
        * (Gammaa ** 2 * Gammab + 2 * Gammab * Oma ** 2 + Gammaa * Omb ** 2)
        * (
            Gammaa * (Gammab * (Gammaa + Gammab) + Oma ** 2)
            + (Gammaa + Gammab) * Omb ** 2
        )
        + Delta ** 2
        * Gammab
        * (
            Gammaa
            * (
                Gammab ** 2 * (2 * Gammaa ** 2 + 2 * Gammaa * Gammab + Gammab ** 2)
                + 2 * Gammab * (Gammaa + 2 * Gammab) * Oma ** 2
                + Oma ** 4
            )
            + Gammab * (Gammaa * (3 * Gammaa + Gammab) + Oma ** 2) * Omb ** 2
            + Gammaa * Omb ** 4
        )
        + delta ** 2
        * (
            Gammaa
            * Gammab
            * (
                Delta ** 2
                + Gammaa ** 2
                + 2 * Gammaa * Gammab
                + 2 * Gammab ** 2
                - 2 * Oma ** 2
            )
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            + (
                Delta ** 2 * Gammaa * (Gammaa + 5 * Gammab)
                + Gammaa ** 2 * (Gammaa ** 2 + Gammaa * Gammab + 2 * Gammab ** 2)
                + 2 * (Gammaa + Gammab) ** 2 * Oma ** 2
            )
            * Omb ** 2
            + Gammaa * Gammab * Omb ** 4
        )
        + 2
        * delta
        * Delta
        * (
            Gammaa
            * Gammab
            * (-(Gammab ** 2) + Oma ** 2)
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            - Gammab
            * (
                Gammaa * (Delta ** 2 + Gammaa ** 2 + 4 * Gammaa * Gammab + Gammab ** 2)
                + Gammab * Oma ** 2
            )
            * Omb ** 2
            - Gammaa * (Gammaa + 2 * Gammab) * Omb ** 4
        )
    )


def s_22(Oma, Omb, Gammaa, Gammab, delta, Delta):
    return (
        Oma ** 2
        * (
            Gammaa
            * Gammab
            * (
                (delta ** 2 + Gammab ** 2)
                * ((delta - Delta) ** 2 + (Gammaa + Gammab) ** 2)
                + 2 * (delta * (-delta + Delta) + Gammab * (Gammaa + Gammab)) * Oma ** 2
                + Oma ** 4
            )
            + (
                -2 * delta * Delta * Gammab ** 2
                + delta ** 2 * (Gammaa ** 2 + Gammaa * Gammab + Gammab ** 2)
                + Gammab
                * (
                    Delta ** 2 * Gammab
                    + (2 * Gammaa + Gammab) * (Gammab * (Gammaa + Gammab) + Oma ** 2)
                )
            )
            * Omb ** 2
            + Gammab * (Gammaa + Gammab) * Omb ** 4
        )
    ) / (
        Delta ** 4 * Gammaa * Gammab ** 3
        + delta ** 4 * Gammaa * Gammab * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
        - 2
        * delta ** 3
        * Delta
        * Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2 + Omb ** 2)
        + (Gammab * (Gammaa + Gammab) + Oma ** 2 + Omb ** 2)
        * (Gammaa ** 2 * Gammab + 2 * Gammab * Oma ** 2 + Gammaa * Omb ** 2)
        * (
            Gammaa * (Gammab * (Gammaa + Gammab) + Oma ** 2)
            + (Gammaa + Gammab) * Omb ** 2
        )
        + Delta ** 2
        * Gammab
        * (
            Gammaa
            * (
                Gammab ** 2 * (2 * Gammaa ** 2 + 2 * Gammaa * Gammab + Gammab ** 2)
                + 2 * Gammab * (Gammaa + 2 * Gammab) * Oma ** 2
                + Oma ** 4
            )
            + Gammab * (Gammaa * (3 * Gammaa + Gammab) + Oma ** 2) * Omb ** 2
            + Gammaa * Omb ** 4
        )
        + delta ** 2
        * (
            Gammaa
            * Gammab
            * (
                Delta ** 2
                + Gammaa ** 2
                + 2 * Gammaa * Gammab
                + 2 * Gammab ** 2
                - 2 * Oma ** 2
            )
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            + (
                Delta ** 2 * Gammaa * (Gammaa + 5 * Gammab)
                + Gammaa ** 2 * (Gammaa ** 2 + Gammaa * Gammab + 2 * Gammab ** 2)
                + 2 * (Gammaa + Gammab) ** 2 * Oma ** 2
            )
            * Omb ** 2
            + Gammaa * Gammab * Omb ** 4
        )
        + 2
        * delta
        * Delta
        * (
            Gammaa
            * Gammab
            * (-(Gammab ** 2) + Oma ** 2)
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            - Gammab
            * (
                Gammaa * (Delta ** 2 + Gammaa ** 2 + 4 * Gammaa * Gammab + Gammab ** 2)
                + Gammab * Oma ** 2
            )
            * Omb ** 2
            - Gammaa * (Gammaa + 2 * Gammab) * Omb ** 4
        )
    )


def s_11(Oma, Omb, Gammaa, Gammab, delta, Delta):
    return (
        delta ** 4 * Gammaa * Gammab * (Delta ** 2 + Gammaa ** 2 + Oma ** 2)
        + Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + Oma ** 2)
        * (Delta ** 2 * Gammab ** 2 + (Gammab * (Gammaa + Gammab) + Oma ** 2) ** 2)
        + Gammab
        * (
            Delta ** 2 * Gammaa * (3 * Gammaa * Gammab + Gammab ** 2 - Oma ** 2)
            + (Gammab * (Gammaa + Gammab) + Oma ** 2)
            * (Gammaa ** 2 * (3 * Gammaa + 2 * Gammab) + (Gammaa + Gammab) * Oma ** 2)
        )
        * Omb ** 2
        + Gammaa
        * (
            Gammab * (Delta ** 2 + (Gammaa + Gammab) * (3 * Gammaa + Gammab))
            + Gammaa * Oma ** 2
        )
        * Omb ** 4
        + Gammaa * (Gammaa + Gammab) * Omb ** 6
        - 2
        * delta ** 3
        * Delta
        * Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + Oma ** 2 + Omb ** 2)
        + delta ** 2
        * (
            Gammaa
            * Gammab
            * (
                Delta ** 2
                + Gammaa ** 2
                + 2 * Gammaa * Gammab
                + 2 * Gammab ** 2
                - 2 * Oma ** 2
            )
            * (Delta ** 2 + Gammaa ** 2 + Oma ** 2)
            + (
                Delta ** 2 * Gammaa * (Gammaa + 5 * Gammab)
                + Gammaa ** 2 * (Gammaa ** 2 + Gammaa * Gammab + 2 * Gammab ** 2)
                + (Gammaa + Gammab) ** 2 * Oma ** 2
            )
            * Omb ** 2
            + Gammaa * Gammab * Omb ** 4
        )
        + 2
        * delta
        * Delta
        * Gammaa
        * (
            Gammab * (Delta ** 2 + Gammaa ** 2 + Oma ** 2) * (-(Gammab ** 2) + Oma ** 2)
            - Gammab
            * (Delta ** 2 + Gammaa ** 2 + 4 * Gammaa * Gammab + Gammab ** 2 - Oma ** 2)
            * Omb ** 2
            - (Gammaa + 2 * Gammab) * Omb ** 4
        )
    ) / (
        Delta ** 4 * Gammaa * Gammab ** 3
        + delta ** 4 * Gammaa * Gammab * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
        - 2
        * delta ** 3
        * Delta
        * Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2 + Omb ** 2)
        + (Gammab * (Gammaa + Gammab) + Oma ** 2 + Omb ** 2)
        * (Gammaa ** 2 * Gammab + 2 * Gammab * Oma ** 2 + Gammaa * Omb ** 2)
        * (
            Gammaa * (Gammab * (Gammaa + Gammab) + Oma ** 2)
            + (Gammaa + Gammab) * Omb ** 2
        )
        + Delta ** 2
        * Gammab
        * (
            Gammaa
            * (
                Gammab ** 2 * (2 * Gammaa ** 2 + 2 * Gammaa * Gammab + Gammab ** 2)
                + 2 * Gammab * (Gammaa + 2 * Gammab) * Oma ** 2
                + Oma ** 4
            )
            + Gammab * (Gammaa * (3 * Gammaa + Gammab) + Oma ** 2) * Omb ** 2
            + Gammaa * Omb ** 4
        )
        + delta ** 2
        * (
            Gammaa
            * Gammab
            * (
                Delta ** 2
                + Gammaa ** 2
                + 2 * Gammaa * Gammab
                + 2 * Gammab ** 2
                - 2 * Oma ** 2
            )
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            + (
                Delta ** 2 * Gammaa * (Gammaa + 5 * Gammab)
                + Gammaa ** 2 * (Gammaa ** 2 + Gammaa * Gammab + 2 * Gammab ** 2)
                + 2 * (Gammaa + Gammab) ** 2 * Oma ** 2
            )
            * Omb ** 2
            + Gammaa * Gammab * Omb ** 4
        )
        + 2
        * delta
        * Delta
        * (
            Gammaa
            * Gammab
            * (-(Gammab ** 2) + Oma ** 2)
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            - Gammab
            * (
                Gammaa * (Delta ** 2 + Gammaa ** 2 + 4 * Gammaa * Gammab + Gammab ** 2)
                + Gammab * Oma ** 2
            )
            * Omb ** 2
            - Gammaa * (Gammaa + 2 * Gammab) * Omb ** 4
        )
    )


def s_31(Oma, Omb, Gammaa, Gammab, delta, Delta):
    return (
        Oma
        * Omb
        * (
            delta ** 3 * (Delta - 1j * Gammaa) * Gammaa * Gammab
            - 1j * Delta ** 3 * Gammaa * Gammab ** 2
            - Delta ** 2 * Gammaa * Gammab * (Gammaa * Gammab - Oma ** 2 + Omb ** 2)
            - delta ** 2
            * Gammaa
            * Gammab
            * ((Delta - 1j * Gammaa) * (2 * Delta + 1j * Gammab) + Oma ** 2 + Omb ** 2)
            - 1j
            * Delta
            * Gammaa
            * Gammab
            * (Gammaa + Gammab)
            * (Gammab * (Gammaa + Gammab) + 2 * Oma ** 2 + Omb ** 2)
            - (
                Gammaa * Gammab * (Gammaa + Gammab)
                - Gammab * Oma ** 2
                + Gammaa * Omb ** 2
            )
            * (
                Gammaa * (Gammab * (Gammaa + Gammab) + Oma ** 2)
                + (Gammaa + Gammab) * Omb ** 2
            )
            + delta
            * Gammaa
            * (
                (Delta - 1j * Gammaa)
                * Gammab
                * (Delta ** 2 + 2j * Delta * Gammab + (Gammaa + Gammab) ** 2)
                + 2j * Gammab * (Gammaa + Gammab) * Oma ** 2
                + ((-1j) * Gammaa * (Gammaa + Gammab) + Delta * (Gammaa + 3 * Gammab))
                * Omb ** 2
            )
        )
    ) / (
        Delta ** 4 * Gammaa * Gammab ** 3
        + delta ** 4 * Gammaa * Gammab * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
        - 2
        * delta ** 3
        * Delta
        * Gammaa
        * Gammab
        * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2 + Omb ** 2)
        + (Gammab * (Gammaa + Gammab) + Oma ** 2 + Omb ** 2)
        * (Gammaa ** 2 * Gammab + 2 * Gammab * Oma ** 2 + Gammaa * Omb ** 2)
        * (
            Gammaa * (Gammab * (Gammaa + Gammab) + Oma ** 2)
            + (Gammaa + Gammab) * Omb ** 2
        )
        + Delta ** 2
        * Gammab
        * (
            Gammaa
            * (
                Gammab ** 2 * (2 * Gammaa ** 2 + 2 * Gammaa * Gammab + Gammab ** 2)
                + 2 * Gammab * (Gammaa + 2 * Gammab) * Oma ** 2
                + Oma ** 4
            )
            + Gammab * (Gammaa * (3 * Gammaa + Gammab) + Oma ** 2) * Omb ** 2
            + Gammaa * Omb ** 4
        )
        + delta ** 2
        * (
            Gammaa
            * Gammab
            * (
                Delta ** 2
                + Gammaa ** 2
                + 2 * Gammaa * Gammab
                + 2 * Gammab ** 2
                - 2 * Oma ** 2
            )
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            + (
                Delta ** 2 * Gammaa * (Gammaa + 5 * Gammab)
                + Gammaa ** 2 * (Gammaa ** 2 + Gammaa * Gammab + 2 * Gammab ** 2)
                + 2 * (Gammaa + Gammab) ** 2 * Oma ** 2
            )
            * Omb ** 2
            + Gammaa * Gammab * Omb ** 4
        )
        + 2
        * delta
        * Delta
        * (
            Gammaa
            * Gammab
            * (-(Gammab ** 2) + Oma ** 2)
            * (Delta ** 2 + Gammaa ** 2 + 2 * Oma ** 2)
            - Gammab
            * (
                Gammaa * (Delta ** 2 + Gammaa ** 2 + 4 * Gammaa * Gammab + Gammab ** 2)
                + Gammab * Oma ** 2
            )
            * Omb ** 2
            - Gammaa * (Gammaa + 2 * Gammab) * Omb ** 4
        )
    )


def laser(delta, lw, ext=10):
    step = delta[1] - delta[0]
    l_v = int(lw / step * ext)
    delta = np.arange(-l_v, l_v + 1) * step
    f = np.exp(-(delta ** 2) / (2 * lw ** 2))
    # f = cauchy.pdf(delta, scale=lw)
    return f / sum(f)


def convolve(dat, kernel):
    """simple convolution"""
    return convolve1d(dat, kernel, mode="nearest")


def co_f(delta, Delta, Oma, Omb, delta0):
    delta = -(delta - delta0)
    return np.abs(s_31(Oma, Omb, Gamma1, Gamma2, delta, Delta)) ** 2


def inco_f(delta, Delta, Oma, Omb, delta0):
    delta = -(delta - delta0)
    return s_33(Oma, Omb, Gamma1, Gamma2, delta, Delta)


def single_f(delta, Delta, Oma, Omb, delta0):
    return inco_f(delta, Delta, Oma, Omb, delta0)


def single_lw(delta, Delta, Oma, Omb, delta0, lw):
    return convolve(single_f(delta, Delta, Oma, Omb, delta0), laser(delta, lw))


def pairs_f(delta, Delta, Oma, Omb, delta0):
    return (
        co_f(delta, Delta, Oma, Omb, delta0)
        + inco_f(delta, Delta, Oma, Omb, delta0) ** 2 * DeltaT
    )


def pairs_lw(delta, Delta, Oma, Omb, delta0, lw):
    return convolve(pairs_f(delta, Delta, Oma, Omb, delta0), laser(delta, lw))


def eff(delta, Delta, Oma, Omb, delta0):
    return pairs_f(delta, Delta, Oma, Omb, delta0) / single_f(
        delta, Delta, Oma, Omb, delta0
    )


def eff_lw(delta, Delta, Oma, Omb, delta0, lw):
    return convolve(eff(delta, Delta, Oma, Omb, delta0), laser(delta, lw))


def signal_f(freq, pump_a, pump_b, parvals):
    return (
        parvals["num"]
        * parvals["etas"]
        * single_lw(
            freq,
            parvals["Delta"],
            np.sqrt(pump_a) * parvals["ma"],
            np.sqrt(pump_b) * parvals["mb"],
            parvals["x0"],
            parvals["lw"],
        )
        + parvals["dc_s"]
    )


def idler_f(freq, pump_a, pump_b, parvals):
    return (
        parvals["num"]
        * parvals["etai"]
        * single_lw(
            freq,
            parvals["Delta"],
            np.sqrt(pump_a) * parvals["ma"],
            np.sqrt(pump_b) * parvals["mb"],
            parvals["x0"],
            parvals["lw"],
        )
        + parvals["dc_i"]
    )


def pair_f(freq, pump_a, pump_b, parvals):
    return (
        parvals["num"]
        * parvals["etai"]
        * parvals["etas"]
        * pairs_lw(
            freq,
            parvals["Delta"],
            np.sqrt(pump_a) * parvals["ma"],
            np.sqrt(pump_b) * parvals["mb"],
            parvals["x0"],
            parvals["lw"],
        )
    )


def eff_s_f(freq, pump_a, pump_b, parvals):
    return pair_f(freq, pump_a, pump_b, parvals) / signal_f(
        freq, pump_a, pump_b, parvals
    )


def eff_i_f(freq, pump_a, pump_b, parvals):
    return pair_f(freq, pump_a, pump_b, parvals) / idler_f(
        freq, pump_a, pump_b, parvals
    )

