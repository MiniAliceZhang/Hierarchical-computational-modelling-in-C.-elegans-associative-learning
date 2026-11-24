# Data

The raw tracking data are not included in this repository. They are large (the full `condition_dict.pkl` is about 15 GB) and are maintained by the laboratory.

## Expected file

`condition_dict.pkl` — a Python dictionary:

```python
{
  "3g-1gNa-1h":  [df_0, df_1, ...],
  "3g-1gNa-4h":  [df_0, df_1, ...],
  "1gNa-end(8x)": [df_0, df_1, ...],
}
```

Each DataFrame is one tracking file and contains at least:

| Column | Meaning |
|---|---|
| `X`, `Y` | pixel coordinates |
| `ID` | worm/trajectory identifier |
| `Timestamp` | frame index at 20 Hz |
| `speed` | linear speed |
| `agl_velocity` | angular velocity |
| `CTX` | chemotaxis index (cosine between heading and low-salt direction) |

The final thesis uses the first 12 min of the 90-min test recording for fitting and primary statistics. High-salt conditions are used only for descriptive range checks, not for the main fit.

## Conditions

| Code | Label |
|---|---|
| `3g-1gNa-1h` | 1 h low-salt |
| `3g-1gNa-4h` | 4 h low-salt |
| `1gNa-end(8x)` | 8 h alternating low/high salt |

## Availability

The raw data can be requested from the author or the laboratory, subject to the relevant data-use agreement.
