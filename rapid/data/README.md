# Demonstration data

`M06_20260129_L_260129_161328_60s.rhd` is the 60-second, 20 kHz Intan RHD demonstration recording included for public distribution with this repository. It contains 9,375 complete 128-sample data blocks and has SHA-256:

```text
89B7A7B796DCFCEC18C999F416CF9179ABFBB60820CF88121E66FA367D3EFD8C
```

Fetch the complete RHD after cloning with `git lfs pull`. On Windows, verify the input with:

```powershell
Get-FileHash .\data\M06_20260129_L_260129_161328_60s.rhd -Algorithm SHA256
```

`ChannelParam_20260129.json` supplies the demonstration channel configuration and enables 191 of 256 channel identifiers. Its SHA-256 is:

```text
383252E222AC47EB70B176B18B8F39003A85E754F121FAE9A9B0381956B69B90
```

The [main README](../README.md) shows how to run the demonstration. Generated results go to the chosen output directory.
