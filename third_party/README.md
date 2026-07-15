# third_party/

Vendored, unmodified snapshots of public repos this project builds on top of.
Each subdirectory has its own `VENDORED.md` recording the exact source URL,
pinned commit, and license. **Never edit these files in place** — wrap or
subclass from `lcmunet/` instead. Pinning exact commits (rather than tracking
upstream or using git submodules) keeps every baseline reproduction on fixed,
known code, which the Fairness rule (same split/seed/preprocessing/hardware/
code across every compared model) depends on.

| Directory | Source | License | Purpose |
|:--|:--|:--|:--|
| `UltraLight-VM-UNet/` | wurenkai/UltraLight-VM-UNet | MIT | Backbone (§4 of the methodology): conv stages + PVM Layer, `c_list=[8,16,24,32,48,64]` |
| `MALUNet/` | JCruan519/MALUNet | **none found** — see its VENDORED.md | Comparator (§9), local reproduction only |
| `EGE-UNet/` | JCruan519/EGE-UNet | Apache-2.0 | Comparator (§9) |
