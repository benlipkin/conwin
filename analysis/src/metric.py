from sklearn.linear_model import RidgeCV
import brainscore_language.metrics.linear_predictivity.metric as bl_linear_metric
from brainscore_language import metric_registry


def rgcv_linear_regression(xarray_kwargs=None):
    regression = RidgeCV(alphas=[1e-3, 0.01, 0.1, 1, 10, 100])
    xarray_kwargs = xarray_kwargs or {}
    regression = bl_linear_metric.XarrayRegression(regression, **xarray_kwargs)
    return regression


def rgcv_linear_pearsonr(
    *args, regression_kwargs=None, correlation_kwargs=None, **kwargs
):
    regression = rgcv_linear_regression(regression_kwargs or {})
    correlation = bl_linear_metric.pearsonr_correlation(correlation_kwargs or {})
    return bl_linear_metric.CrossRegressedCorrelation(
        *args, regression=regression, correlation=correlation, **kwargs
    )


metric_registry["rgcv_linear_pearsonr"] = rgcv_linear_pearsonr
