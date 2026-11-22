"""Un GP aditivo regional/local con varianza total fija; hipótesis experimental."""

from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def multiscale_kernel(config):
    """Covarianza conjunta, no promedio de dos posteriores ajustados por separado."""
    if (
        config.x_length_scale_m != 2000
        or config.y_length_scale_m != 2000
        or config.temporal_length_scale_s != 3600
    ):
        raise ValueError("El control multiescala requiere GP original 2000 m/3600 s")
    regional = Matern([2000, 2000, 3600], length_scale_bounds="fixed", nu=1.5)
    local = Matern([500, 500, 3600], length_scale_bounds="fixed", nu=1.5)
    return ConstantKernel(
        config.prior_stddev_ug_m3**2 / 2, constant_value_bounds="fixed"
    ) * (regional + local)
