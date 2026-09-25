"""Structural pose optimization (StructReg).

UAV geo-localization posed as a global optimization over the 2-D similarity
group Sim(2): find the pose T=(s, theta, t) that best places the UAV's
building structure on the satellite map's building structure.

    J(T) = sum_b w_b G_S(T x_b)                 (smoothed chamfer / edge term)
         + alpha <M_U o T^-1, M_S>             (signed-mask overlap term)
         - (theta-theta0)^2/2 s_th^2 - (log s - log s0)^2/2 s_s^2   (priors)

Modules
-------
structure   probability map -> masks, boundary points, persistence weights,
            satellite kernel / signed-mask maps.
fft_search  exhaustive search over a (theta, s) grid; every translation is
            scored at once by FFT cross-correlation (exact on the grid).
refine      continuous refinement of J and Laplace covariance of the pose.
verify      Ekeland / MFCA building-shape agreement of a pose hypothesis.
integrity   integrity features, calibration, risk-coverage metrics.
"""
