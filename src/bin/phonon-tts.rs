use anyhow::Result;
use catena_models::phonon::{DEFAULT_MODEL, DEFAULT_VOICE, Phonon};
use clap::{Parser, ValueEnum};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(about = "Gradium Phonon text-to-speech through Catena (24 kHz WAV)")]
struct Args {
    text: String,
    /// Hugging Face model ID (defaults to the English checkpoint)
    #[arg(long, value_name = "MODEL")]
    model: Option<String>,
    /// Select a language checkpoint, or use the multilingual checkpoint with auto
    #[arg(long, value_enum, conflicts_with = "model")]
    language: Option<Language>,
    /// Voice name as written in the Gradium voice catalog
    #[arg(long, default_value = DEFAULT_VOICE, value_name = "NAME")]
    voice: String,
    #[arg(long, default_value = "speech.wav")]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    timings: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Language {
    Auto,
    En,
    Fr,
    De,
    Pt,
    Es,
}

impl Language {
    fn model(self) -> &'static str {
        match self {
            Self::Auto => "gradium/phonon-40m-multilingual",
            Self::En => DEFAULT_MODEL,
            Self::Fr => "gradium/phonon-40m-fr",
            Self::De => "gradium/phonon-40m-de",
            Self::Pt => "gradium/phonon-40m-pt",
            Self::Es => "gradium/phonon-40m-es",
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        !args.text.trim().is_empty(),
        "text must contain a spoken prompt"
    );
    let start = Instant::now();
    let checkpoint = args
        .model
        .as_deref()
        .or_else(|| args.language.map(Language::model))
        .unwrap_or(DEFAULT_MODEL);
    let model = Phonon::load_model_with_voice(checkpoint, &args.voice)?;
    if args.timings {
        eprintln!("load and compile {:.3}s", start.elapsed().as_secs_f64());
    }
    model
        .synthesize(&args.text, args.seed, args.timings)?
        .write_wav(&args.output)?;
    println!("{}", args.output.display());
    Ok(())
}
