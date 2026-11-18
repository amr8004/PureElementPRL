"""
Liquid-phase heat capacity (Cp) models for pure elements.

These mirror the solid-phase models already fit via MCMC in this repo
(``RWModelE``, ``CSModelE``, ``SRModelE`` in ``pycalphad.model``, dispatched
by ``espei.pureElement.select_model``/``log_prior``): every function here has
the same ``f(x, *param) -> Cp(T)`` signature, so it drops into
``espei.pureElement.Cp_fit`` / ``MCMC_fit`` unchanged. They are kept in this
repo for local testing; once validated, move the function body into
``pycalphad/model.py`` (reusing ``Einstein``/``RTDB_globals`` already defined
there) and add the ``model_flag`` branches to
``espei/pureElement.py:select_model`` and ``log_prior`` (see
``LIQUID_MODEL_REGISTRY`` / ``LIQUID_MODEL_PRIORS`` below for the values to
port into those two functions).

Models implemented, roughly in order of increasing physical detail:

1. ``DulongPetitLiquidE``  -- classical high-T limit Cp = 3R + linear term.
   Crude baseline; useful as a sanity check / fallback.
2. ``KelleyLiquidE``       -- Kelley, K.K. (1960) "Contributions to the Data
   on Theoretical Metallurgy", US Bureau of Mines Bulletin 584.
   Cp = a + bT + c/T^2. The classical pre-CALPHAD empirical fit for liquid
   metals, still widely quoted (e.g. Kubaschewski, Alcock & Spencer,
   *Materials Thermochemistry*).
3. ``SGTELiquidE``         -- Dinsdale, A.T. (1991) "SGTE data for pure
   elements", Calphad 15(4), 317-425. Cp derived from the standard CALPHAD
   unary Gibbs energy polynomial G = a + bT + cT*ln(T) + dT^2 + eT^-1 + fT^3,
   i.e. Cp = -c - 2dT - 2e/T^2 - 6fT^2. This is the de facto standard liquid
   description in SGTE-compatible unary databases.
4. ``EinsteinLinearLiquidE`` / ``EinsteinQuarticLiquidE`` -- Einstein
   oscillator (same ``Einstein()`` helper used by the solid models) plus a
   linear/quadratic or quartic polynomial, mirroring ``model_RWE`` and
   ``model_CS`` (Chen, Q. & Sundman, B. (2001) Acta Materialia 49, 947-961)
   but applied above Tm, since the short-range vibrational contribution does
   not vanish immediately on melting.
5. ``TwoStateLiquidE``     -- phenomenological version of the "two-state"
   liquid model (Agren, J. (1979) Scripta Metallurgica 13, 535-541; extended
   to many pure elements by Bigdeli, S., Chen, Q. & Selleby, M. (2016) Calphad
   53, 44-49, and Bigdeli et al. (2018) J. Phase Equilib. Diffus.). The real
   two-state model gets its Cp bump near Tm from minimizing a Gibbs energy
   that mixes a solid-like and a liquid-like (disordered) state; that
   minimization is not reproduced here. Instead the same qualitative
   signature -- a symmetric excess-Cp bump of height ``A`` centered on a
   transition temperature ``Tt`` with width ``w`` -- is added on top of an
   Einstein baseline, as the derivative of a logistic fraction-disordered
   curve. Treat this as a fitting-convenient stand-in for the real two-state
   model, not a re-derivation of it.

Every model also gets an entry in ``LIQUID_MODEL_REGISTRY`` (name ->
function), ``LIQUID_MODEL_PARAMS`` (name -> ordered parameter names) and
``LIQUID_MODEL_PRIORS`` (name -> flat-prior bounds per parameter, in the same
style as ``espei.pureElement.log_prior``) so the whole set can be driven the
same way as the existing solid models.
"""
import numpy as np
from pycalphad.model import Einstein

R = 8.314


def DulongPetitLiquidE(x, *param):
    """Cp = 3R + a*T. Classical high-temperature (equipartition) limit."""
    a = param[0]
    x = np.asarray(x, dtype=float)
    return 3.0 * R + a * x


def KelleyLiquidE(x, *param):
    """Kelley (1960): Cp = a + b*T + c/T^2."""
    a, b, c = param[0], param[1], param[2]
    x = np.asarray(x, dtype=float)
    return a + b * x + c / x**2


def SGTELiquidE(x, *param):
    """Dinsdale (1991) SGTE unary polynomial: Cp = p0 + p1*T + p2/T^2 + p3*T^2,
    i.e. -c, -2d, -2e, -6f from G = a + bT + cT*ln(T) + dT^2 + eT^-1 + fT^3."""
    p0, p1, p2, p3 = param[0], param[1], param[2], param[3]
    x = np.asarray(x, dtype=float)
    return p0 + p1 * x + p2 / x**2 + p3 * x**2


def EinsteinLinearLiquidE(x, *param):
    """Einstein oscillator + linear/quadratic polynomial, mirroring
    pycalphad.model.model_RWE applied to the liquid."""
    Theta_E, a, b = param[0], param[1], param[2]
    x = np.asarray(x, dtype=float)
    return Einstein(Theta_E, x) + a * x + b * x**2


def EinsteinQuarticLiquidE(x, *param):
    """Einstein oscillator + T^4 term, mirroring pycalphad.model.model_CS
    (Chen & Sundman, 2001) applied to the liquid."""
    Theta_E, a, b = param[0], param[1], param[2]
    x = np.asarray(x, dtype=float)
    return Einstein(Theta_E, x) + a * x + b * x**4


def TwoStateLiquidE(x, *param):
    """Einstein baseline + a symmetric excess-Cp bump (derivative of a
    logistic "fraction disordered" curve), as a fitting-convenient stand-in
    for the Agren (1979) / Bigdeli et al. (2016) two-state liquid model.

    param: Theta_E, a, A, Tt, w
        Theta_E -- Einstein temperature of the harmonic baseline
        a       -- linear coefficient of the harmonic baseline
        A       -- amplitude of the excess-Cp bump
        Tt      -- temperature at which the bump is centered (~ near Tm)
        w       -- width of the bump
    """
    Theta_E, a, A, Tt, w = param[0], param[1], param[2], param[3], param[4]
    x = np.asarray(x, dtype=float)
    z = (x - Tt) / w
    excess = A * np.exp(z) / (w * (1.0 + np.exp(z)) ** 2)
    return Einstein(Theta_E, x) + a * x + excess


LIQUID_MODEL_REGISTRY = {
    'DulongPetitLiquidE': DulongPetitLiquidE,
    'KelleyLiquidE': KelleyLiquidE,
    'SGTELiquidE': SGTELiquidE,
    'EinsteinLinearLiquidE': EinsteinLinearLiquidE,
    'EinsteinQuarticLiquidE': EinsteinQuarticLiquidE,
    'TwoStateLiquidE': TwoStateLiquidE,
}

LIQUID_MODEL_PARAMS = {
    'DulongPetitLiquidE': ['a'],
    'KelleyLiquidE': ['a', 'b', 'c'],
    'SGTELiquidE': ['p0', 'p1', 'p2', 'p3'],
    'EinsteinLinearLiquidE': ['Theta_E', 'a', 'b'],
    'EinsteinQuarticLiquidE': ['Theta_E', 'a', 'b'],
    'TwoStateLiquidE': ['Theta_E', 'a', 'A', 'Tt', 'w'],
}

# Flat-prior bounds (min, max) per parameter, in the same spirit as the
# hardcoded ranges in espei.pureElement.log_prior. These are starting points,
# not physical constants -- tighten them once you have real liquid Cp data
# for a given element.
LIQUID_MODEL_PRIORS = {
    'DulongPetitLiquidE': [(-0.1, 0.1)],
    'KelleyLiquidE': [(-50.0, 50.0), (-0.1, 0.1), (-1e6, 1e6)],
    'SGTELiquidE': [(-100.0, 100.0), (-0.5, 0.5), (-1e7, 1e7), (-1e-3, 1e-3)],
    'EinsteinLinearLiquidE': [(0.0, 700.0), (0.0, 0.1), (0.0, 0.1)],
    'EinsteinQuarticLiquidE': [(0.0, 700.0), (0.0, 0.1), (0.0, 0.1)],
    'TwoStateLiquidE': [(0.0, 700.0), (0.0, 0.1), (-500.0, 500.0), (200.0, 3500.0), (1.0, 500.0)],
}


# Generic starting points for curve_fit / MCMC initialization. These are not
# tuned to any particular element -- they just land each model in a plausible
# liquid-metal Cp range (roughly 25-35 J/K/mol) over a few hundred to a couple
# thousand K, so a nonlinear fit has somewhere reasonable to start from.
LIQUID_MODEL_INITIAL_GUESS = {
    'DulongPetitLiquidE': [1e-3],
    'KelleyLiquidE': [25.0, 5e-3, -1e5],
    'SGTELiquidE': [25.0, 5e-3, -1e5, -1e-6],
    'EinsteinLinearLiquidE': [300.0, 1e-3, 1e-6],
    'EinsteinQuarticLiquidE': [300.0, 1e-3, 1e-11],
    'TwoStateLiquidE': [300.0, 1e-3, 15.0, 1300.0, 80.0],
}


def select_liquid_model(model_flag):
    """Mirrors espei.pureElement.select_model for the liquid-model registry."""
    try:
        return LIQUID_MODEL_REGISTRY[model_flag]
    except KeyError:
        raise ValueError("Invalid model_flag: %r" % (model_flag,))


def liquid_log_prior(param, model_flag):
    """Flat prior within LIQUID_MODEL_PRIORS[model_flag], mirroring
    espei.pureElement.log_prior."""
    bounds = LIQUID_MODEL_PRIORS[model_flag]
    for value, (lo, hi) in zip(param, bounds):
        if not (lo < value < hi):
            return -np.inf
    return 0.0


if __name__ == '__main__':
    # Smoke test: every model should evaluate without error over a plausible
    # liquid-range temperature grid. This does NOT validate the fit against
    # real data -- see the module docstring for how to wire these into
    # espei.pureElement.Cp_fit / MCMC_fit with an element's real Cp data.
    T = np.linspace(600.0, 2200.0, 50)
    for name, func in LIQUID_MODEL_REGISTRY.items():
        cp = func(T, *LIQUID_MODEL_INITIAL_GUESS[name])
        assert np.all(np.isfinite(cp)), f"{name} produced non-finite Cp"
        print(f"{name}: Cp(T) range = [{np.min(cp):.3f}, {np.max(cp):.3f}] J/K/mol")
