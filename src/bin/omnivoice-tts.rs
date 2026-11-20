use anyhow::{Result, ensure};
use catena_models::omnivoice::OmniVoice;
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "OmniVoice text-to-speech through Catena (24 kHz WAV)")]
struct Args {
    text: String,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    timings: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.text.trim().is_empty(),
        "text must contain a spoken prompt"
    );
    let start = Instant::now();
    let model = OmniVoice::load()?;
    if args.timings {
        eprintln!("load and compile {:.3}s", start.elapsed().as_secs_f64());
    }
    let start = Instant::now();
    let audio = model.synthesize(&args.text, args.seed)?;
    if args.timings {
        eprintln!("synthesis {:.3}s", start.elapsed().as_secs_f64());
    }
    audio.write_wav(&args.output)?;
    println!("{}", args.output.display());
    Ok(())
}
