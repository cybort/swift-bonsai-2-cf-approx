# ptq1_dequant_cli

Small Rust CLI that dequantizes raw **PTQ1_0** (28-byte blocks) dumps to f32.

Algorithm matches Prism / rlx-gguf `ptq1_dequant`. The default production path is the Python package `swift_cf_approx.dequant`.

```bash
cargo build --manifest-path crates/ptq1_dequant_cli/Cargo.toml
./crates/ptq1_dequant_cli/target/debug/ptq1_dequant_cli --input packed.bin --output out.f32
```
