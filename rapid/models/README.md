# Fixed demonstration reference map

`fpga14_local_peak_no_boundary_tmpl0129.csv` is the fixed direct-assignment map used by the demonstration. It contains 591 non-residual intervals across 191 channels and both polarities. Each channel/polarity group is contiguous, with its outer interval extending to negative or positive infinity.

The map contains the seven fields used during replay:

```text
unit_id, channel, polarity, unit_index, is_residual, lower_uv, upper_uv
```

Its SHA-256 is:

```text
DD7B69BDC08EFE9ABE9F7A2B48DD511C0B6133C8DA6B6429525582CA5A16D984
```

The demo loads this map directly; no model fitting is required. Interval conventions and the optional PCA command are explained in [docs/METHODS.md](../docs/METHODS.md).
