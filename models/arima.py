"""Monthly residual ARIMA using Burg estimation and the existing forecast mapping."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ArimaResidualModel:
    """ARIMA on monthly medians of remaining log-price residuals."""

    def __init__(self, order=(1, 1, 0)):
        self.order = order
        self.result = None
        self.first_month = None
        self.last_month = None

    @staticmethod
    def _months(dates):
        parsed = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
        if parsed.isna().any():
            raise ValueError("ARIMA needs valid date_listed values")
        return parsed.dt.to_period("M")

    def fit(self, dates, log_residuals):
        months = self._months(dates)
        residuals = np.asarray(log_residuals, dtype=float).reshape(-1)
        if len(months) != len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("ARIMA dates and residuals must be aligned and finite")

        monthly = pd.Series(residuals).groupby(months).median().sort_index()
        if len(monthly) < 6:
            raise ValueError("ARIMA needs at least six historical months")

        # Build a regular monthly time series. Missing months (if any) are
        # interpolated using only the training-period residual series.
        month_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
        monthly = monthly.reindex(month_index).interpolate(limit_direction="both")
        self.first_month, self.last_month = month_index[0], month_index[-1]
        self.result = ARIMA(monthly, order=self.order).fit()

        details = getattr(self.result, "mle_retvals", None)
        if details is not None and not details.get("converged", False):
            print({
                "order": self.order,
                "months": len(monthly),
                "period": (str(self.first_month), str(self.last_month)),
                "warnflag": details.get("warnflag"),
                "iterations": details.get("iterations"),
                "gradient": details.get("gopt"),
                "parameters": self.result.params.to_dict(),
            })
        return self

    def predict(self, dates):
        if self.result is None:
            raise RuntimeError("Fit ARIMA before predicting")
        months = self._months(dates)
        if months.empty:
            return np.array([], dtype=float)
        offsets = months.astype("int64").to_numpy() - self.first_month.ordinal
        if (offsets < 0).any():
            raise ValueError("Cannot predict before the first ARIMA training month")
        # In-sample predictions are one-step-ahead fitted values; future
        # months use forecasts. One call handles both and preserves row order.
        values = np.asarray(
            self.result.predict(start=0, end=int(offsets.max())), dtype=float
        )
        return values[offsets]

