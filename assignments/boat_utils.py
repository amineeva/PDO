__all__ = [
    "make_hull",
    "make_hull_box",
    "make_hull_tall",
    "make_hull_wide",
    "hull_data",
    "hull_rotate",
    "get_avs",
    "get_mass_properties",
    "get_moment_curve",
    "get_buoyant_properties",
    "get_equ_waterline",
    "RHO_WATER",
]

from numpy import array, average, concatenate, linspace, meshgrid, sum, min
from numpy import sin, cos, pi, NaN
from pandas import concat, DataFrame
from scipy.optimize import bisect

# Global constants
RHO_WATER = 0.03613 # Density of water (lb / in^3)
RHO_0 = 0.04516 # Filament density (lb / in^3)
G = 386 # Gravitational acceleration (in / s^2)

## Boat generator
# --------------------------------------------------
def make_hull(X):
    r"""

    Args:
        X (iterable): [w_k, h_k, w_h] = X

        w_k = sharpness of keel, w_k \in [0.2, 0.4]
        h_k = height of keel, h_k \in [1.0, 2.0]
        w_h = sharpness of hull, w_h \in [1.0, 3.0]

    Returns:
        DataFrame: Hull points
        DataFrame: Mass properties
    """
    w_k, h_k, w_h = X

    f_hull = lambda x: min([
        (x / w_k)**2,      # Keel
        (x / w_h)**2 + h_k # Hull
    ])
    g_top = lambda x, y: y <= 3
    rho_hull = lambda x, y: RHO_0 * (y <= h_k) + 0.25 * RHO_0 * (y > h_k)
    f_hull = lambda x: min([
        (x / w_k)**2,      # Keel
        (x / w_h)**2 + h_k # Hull
    ])

    df_hull, dx, dy = hull_data(
        f_hull,
        g_top,
        rho_hull,
        n_marg=50,
    )

    df_mass = get_mass_properties(df_hull, dx, dy)

    return df_hull, df_mass

def make_hull_box(n_marg=50):
    r"""

    Returns
        DataFrame: Hull points
        DataFrame: Mass properties

    """
    Xv = linspace(-1, +1, num=n_marg)
    Yv = linspace(-1, +1, num=n_marg)
    dx = Xv[1] - Xv[0]
    dy = Yv[1] - Yv[0]

    Xm, Ym = meshgrid(Xv, Yv)
    n_tot = Xm.shape[0] * Xm.shape[1]

    Z = concatenate(
        (Xm.reshape(n_tot, -1), Ym.reshape(n_tot, -1)),
        axis=1,
    )

    M = array([0.25 * RHO_WATER * dx * dy for x, y in Z])

    Z_hull = Z
    M_hull = M

    df_hull = DataFrame(dict(
        x=Z_hull[:, 0],
        y=Z_hull[:, 1],
        dm=M_hull,
    ))

    df_mass = get_mass_properties(df_hull, dx, dy)

    return df_hull, df_mass

def make_hull_tall(n_marg=50):
    r"""

    Returns
        DataFrame: Hull points
        DataFrame: Mass properties

    """
    Xv = linspace(-1, +1, num=n_marg)
    Yv = linspace(-3, +3, num=n_marg)
    dx = Xv[1] - Xv[0]
    dy = Yv[1] - Yv[0]

    Xm, Ym = meshgrid(Xv, Yv)
    n_tot = Xm.shape[0] * Xm.shape[1]

    Z = concatenate(
        (Xm.reshape(n_tot, -1), Ym.reshape(n_tot, -1)),
        axis=1,
    )

    M = array([0.25 * RHO_WATER * dx * dy for x, y in Z])

    Z_hull = Z
    M_hull = M

    df_hull = DataFrame(dict(
        x=Z_hull[:, 0],
        y=Z_hull[:, 1],
        dm=M_hull,
    ))

    df_mass = get_mass_properties(df_hull, dx, dy)

    return df_hull, df_mass

def make_hull_wide(n_marg=50):
    r"""

    Returns
        DataFrame: Hull points
        DataFrame: Mass properties

    """
    Xv = linspace(-3, +3, num=n_marg)
    Yv = linspace(-1, +1, num=n_marg)
    dx = Xv[1] - Xv[0]
    dy = Yv[1] - Yv[0]

    Xm, Ym = meshgrid(Xv, Yv)
    n_tot = Xm.shape[0] * Xm.shape[1]

    Z = concatenate(
        (Xm.reshape(n_tot, -1), Ym.reshape(n_tot, -1)),
        axis=1,
    )

    M = array([0.25 * RHO_WATER * dx * dy for x, y in Z])

    Z_hull = Z
    M_hull = M

    df_hull = DataFrame(dict(
        x=Z_hull[:, 0],
        y=Z_hull[:, 1],
        dm=M_hull,
    ))

    df_mass = get_mass_properties(df_hull, dx, dy)

    return df_hull, df_mass

## Hull manipulation
# --------------------------------------------------
def hull_data(f_hull, g_top, rho_hull, n_marg=50, x_wid=2, y_lo=+0, y_hi=+4):
    r"""
    Args:
        f_hull (lambda): Function of signature y = f(x);
            defines lower surface of boat
        g_top (lambda): Function of signature g (bool) = g(x, y);
            True indicates within boat
        rho_hull (lambda): Function of signature rho = rho(x, y);
            returns local hull density

    Returns:
        DataFrame: x, y, dm boat hull points and element masses
        float: dx
        float: dy
    """
    Xv = linspace(-x_wid, +x_wid, num=n_marg)
    Yv = linspace(y_lo, y_hi, num=n_marg)
    dx = Xv[1] - Xv[0]
    dy = Yv[1] - Yv[0]

    Xm, Ym = meshgrid(Xv, Yv)
    n_tot = Xm.shape[0] * Xm.shape[1]

    Z = concatenate(
        (Xm.reshape(n_tot, -1), Ym.reshape(n_tot, -1)),
        axis=1,
    )

    M = array([rho_hull(x, y) * dx * dy for x, y in Z])

    I_hull = [
        (f_hull(x) <= y) & g_top(x, y)
        for x, y in Z
    ]
    Z_hull = Z[I_hull]
    M_hull = M[I_hull]

    df_hull = DataFrame(dict(
        x=Z_hull[:, 0],
        y=Z_hull[:, 1],
        dm=M_hull,
    ))

    return df_hull, dx, dy

def hull_rotate(df_hull, df_mass, angle):
    r"""
    Args:
        df_hull (DataFrame): Hull points
        df_mass (DataFrame): Mass properties, gives COM
        angle (float, radians): Heel angle

    Returns:
        DataFrame: Hull points rotated about COM
    """
    R = array([
        [cos(angle), -sin(angle)],
        [sin(angle),  cos(angle)]
    ])
    Z_hull_r = (
        df_hull[["x", "y"]].values - df_mass[["x", "y"]].values
    ).dot(R.T) + df_mass[["x", "y"]].values

    return DataFrame(dict(
        x=Z_hull_r[:, 0],
        y=Z_hull_r[:, 1],
        dm=df_hull.dm,
    ))

## Evaluate hull
# --------------------------------------------------
def get_mass_properties(df_hull, dx, dy):
    x_com = average(df_hull.x, weights=df_hull.dm)
    y_com = average(df_hull.y, weights=df_hull.dm)
    mass = df_hull.dm.sum()

    return DataFrame(dict(
        x=[x_com],
        y=[y_com],
        dx=[dx],
        dy=[dy],
        mass=[mass]
    ))

def get_buoyant_properties(df_hull_rot, df_mass, y_water):
    r"""
    Args:
        df_hull_rot (DataFrame): Rotated hull points
        df_mass (DataFrame): Mass properties
        y_water (float): Location of waterline (in absolute coordinate system)
    """
    dx = df_mass.dx[0]
    dy = df_mass.dy[0]

    I_under = df_hull_rot.y <= y_water
    x_cob = average(df_hull_rot[I_under].x)
    y_cob = average(df_hull_rot[I_under].y)

    m_water = RHO_WATER * sum(I_under) * dx * dy
    F_net = (m_water - df_mass.mass[0]) * G
    M_net = G * m_water * (x_cob - df_mass.x[0])
    # Righting moment == opposite true moment?
    ## Could just use moment arm

    return DataFrame(dict(
        x=[x_cob],
        y=[y_cob],
        F_net=[F_net],
        M_net=[M_net],
    ))

def get_equ_waterline(df_hull, df_mass, angle, y_l=1, y_h=4):
    r"""
    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        angle (float): Angle of rotation
        y_l (float): Low-bound for waterline
        y_h (float): High-bound for waterline

    Returns:
        float: Waterline of zero net vertical force (heave-steady)
    """
    dx = df_mass.dx[0]
    dy = df_mass.dy[0]

    df_hull_r = hull_rotate(df_hull, df_mass, angle)

    def fun(y_g):
        df_buoy = get_buoyant_properties(
            df_hull_r,
            df_mass,
            y_g,
        )

        return df_buoy.F_net[0]

    try:
        y_star = bisect(fun, y_l, y_h, maxiter=1000)

        df_res = get_buoyant_properties(
                df_hull_r,
                df_mass,
                y_star,
            )
        df_res["y_w"] = y_star
    except ValueError:
        df_res = DataFrame(dict(M_net=[NaN], y_w=[NaN]))

    return df_res

def get_moment_curve(df_hull, df_mass, a_l=0, a_h=pi, num=50, y_l=1, y_h=4):
    r"""Generate a righting moment curve

    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        a_l (float): Low-bound for angle
        a_h (float): High-bound for angle
        num (int): Number of points to sample (linearly) between a_l, a_h
        y_l (float): Low-bound for waterline
        y_h (float): High-bound for waterline

    Returns:
        DataFrame: Data from angle sweep
    """
    df_res = DataFrame()
    a_all = linspace(a_l, a_h, num=50)

    for angle in a_all:
        df_tmp = get_equ_waterline(df_hull, df_mass, angle, y_l=y_l, y_h=y_h)
        df_tmp["angle"] = angle

        df_res = concat((df_res, df_tmp), axis=0)
    df_res.reset_index(inplace=True, drop=True)

    return df_res

def get_avs(df_hull, df_mass, a_l=0.1, a_h=pi - 0.1):
    r"""
    Args:
        df_hull (DataFrame): Unrotated hull points
        df_mass (DataFrame): Mass properties
        a_l (float): Low-bound for angle
        a_h (float): High-bound for angle

    Returns:
        float: Angle of vanishing stability
    """
    # Create helper function
    def fun(angle):
        df_res = get_equ_waterline(
            df_hull,
            df_mass,
            angle,
        )

        return df_res.M_net[0]

    # Bisect for zero-moment
    try:
        a_star = bisect(fun, a_l, a_h, maxiter=1000)

        df_res = get_equ_waterline(
            df_hull,
            df_mass,
            a_star,
        )
        df_res["angle"] = a_star
    except ValueError:
        df_res = DataFrame(dict(angle=[NaN]))

    return df_res
