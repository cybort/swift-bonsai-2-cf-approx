//! Optional PTQ1_0 block dequant CLI (algorithm aligned with rlx-gguf / Prism).
//! The primary production path is the Python package `swift_cf_approx.dequant`.

use anyhow::{bail, Result};
use clap::Parser;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

const QK: usize = 128;
const QS: usize = 24;
const QH: usize = 2;
const BLOCK: usize = 28;
const STAGES: [usize; 3] = [32, 16, 8];
const POW3: [u16; 5] = [1, 3, 9, 27, 81];

fn trit(byte: u8, n: usize) -> i32 {
    let q = (byte as u16).wrapping_mul(POW3[n]) as u8;
    ((q as u16 * 3) >> 8) as i32 - 1
}

fn dequant_block(block: &[u8], out: &mut [f32; QK]) {
    let qs = &block[0..QS];
    let qh = &block[QS..QS + QH];
    let d = half::f16::from_le_bytes([block[QS + QH], block[QS + QH + 1]]).to_f32();
    let mut o = 0usize;
    let mut j = 0usize;
    for c in STAGES {
        while j + c <= QS {
            for n in 0..5 {
                for m in 0..c {
                    out[o] = trit(qs[j + m], n) as f32 * d;
                    o += 1;
                }
            }
            j += c;
        }
    }
    for n in 0..4 {
        for &h in qh {
            out[o] = trit(h, n) as f32 * d;
            o += 1;
        }
    }
    debug_assert_eq!(o, QK);
}

#[derive(Parser, Debug)]
#[command(about = "Dequant one or more PTQ1_0 blocks from a raw dump to f32 bin")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut raw = Vec::new();
    File::open(&args.input)?.read_to_end(&mut raw)?;
    if raw.len() % BLOCK != 0 {
        bail!("input length {} not multiple of {}", raw.len(), BLOCK);
    }
    let nblocks = raw.len() / BLOCK;
    let mut out = vec![0f32; nblocks * QK];
    for i in 0..nblocks {
        let mut block_out = [0f32; QK];
        dequant_block(&raw[i * BLOCK..(i + 1) * BLOCK], &mut block_out);
        out[i * QK..(i + 1) * QK].copy_from_slice(&block_out);
    }
    let mut f = File::create(&args.output)?;
    for v in out {
        f.write_all(&v.to_le_bytes())?;
    }
    println!("wrote {} f32 values to {}", nblocks * QK, args.output.display());
    Ok(())
}
