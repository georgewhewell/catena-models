use anyhow::Result;
use catena_models::phonon::{DEFAULT_MODEL, Phonon};
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "Gradium Phonon text-to-speech through Catena (24 kHz WAV)")]
struct Args {
    text: String,
    #[arg(long, default_value = DEFAULT_MODEL, value_name = "MODEL")]
    model: String,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    timings: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        !args.text.trim().is_empty(),
        "text must contain a spoken prompt"
    );
    let start = Instant::now();
    let model = Phonon::load_model(&args.model)?;
    if args.timings {
        eprintln!("load and compile {:.3}s", start.elapsed().as_secs_f64());
    }
    model
        .synthesize(&args.text, args.seed, args.timings)?
        .write_wav(&args.output)?;
    println!("{}", args.output.display());
    Ok(())
}
