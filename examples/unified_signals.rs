/*
Stream wav files through ParakeetUnified and print what the decoder committed,
the per-step model signals, the tentative tail of each step and its cost.

cargo run --example unified_signals -- <model_dir> name=a.wav[+b.wav...] ...

Each clip is fed in chunk-sized blocks and flushed. A `+` joins several wavs into
one clip. Output is one JSON object per clip on stdout and a cost summary per
clip on stderr. Token frames are encoder frames (80 ms).
*/

use parakeet_rs::{ParakeetUnified, ParakeetUnifiedHandle, StepStats, TimestampMode};
use std::env;

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return Err(format!("{path}: expected 16kHz, got {}Hz", spec.sample_rate).into());
    }
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<_, _>>()?,
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    Ok(audio)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let (model_dir, clips) = args
        .split_first()
        .ok_or("usage: <model_dir> name=wav ...")?;
    let handle = ParakeetUnifiedHandle::load(model_dir, None)?;

    for clip in clips {
        let (name, spec) = clip.split_once('=').ok_or("clip must be name=wav")?;
        let mut audio = Vec::new();
        for path in spec.split('+') {
            audio.extend(load_wav(path)?);
        }

        let mut model = ParakeetUnified::from_shared(&handle);
        let chunk = model.streaming_config().chunk_samples();
        let mut emitted = String::new();
        let mut steps = Vec::new();
        let mut stats: Vec<StepStats> = Vec::new();
        let mut last = StepStats::default();
        for block in audio.chunks(chunk) {
            emitted.push_str(&model.transcribe_chunk(block)?);
            if model.last_step_stats() != last {
                last = model.last_step_stats();
                stats.push(last);
                let tentative: Vec<serde_json::Value> = model
                    .tentative()
                    .iter()
                    .map(|t| {
                        serde_json::json!([t.text, t.abs_frame, t.frames_to_edge, t.confidence])
                    })
                    .collect();
                steps.push(serde_json::json!({
                    "committed": model.get_transcript(),
                    "with_tentative": model.transcript_with_tentative(),
                    "tentative": tentative,
                }));
            }
        }
        emitted.push_str(&model.flush()?);
        stats.push(model.last_step_stats());

        let result = model.get_timed_transcript(TimestampMode::Tokens);
        let tokens: Vec<serde_json::Value> = result
            .tokens
            .iter()
            .zip(model.token_confidences())
            .map(|(t, c)| {
                serde_json::json!([t.text, t.start.to_bits(), t.end.to_bits(), c.to_bits()])
            })
            .collect();
        let committed: Vec<serde_json::Value> = result
            .tokens
            .iter()
            .map(|t| serde_json::json!([t.text, (t.start / 0.08).round() as usize]))
            .collect();
        let details: Vec<serde_json::Value> = model
            .token_details()
            .iter()
            .map(|d| {
                serde_json::json!({
                    "top": d.top, "logp_blank": d.logp_blank, "entropy": d.entropy,
                    "tsallis_confidence": d.tsallis_confidence, "margin": d.margin,
                })
            })
            .collect();
        let frames: Vec<serde_json::Value> = model
            .frame_signals()
            .iter()
            .map(|f| {
                serde_json::json!({
                    "abs_frame": f.abs_frame, "log10_nonblank": f.log10_nonblank, "top": f.top,
                })
            })
            .collect();
        let us = |f: fn(&StepStats) -> f64| -> (f64, f64) {
            let v: Vec<f64> = stats.iter().map(f).collect();
            let max = v.iter().copied().fold(0.0, f64::max);
            (v.iter().sum::<f64>() / v.len().max(1) as f64, max)
        };
        let (enc_mean, enc_max) = us(|s| s.encoder.as_secs_f64() * 1e3);
        let (dec_mean, dec_max) = us(|s| s.decode.as_secs_f64() * 1e3);
        let (sig_mean, sig_max) = us(|s| s.signals.as_secs_f64() * 1e3);
        let (ten_mean, ten_max) = us(|s| s.tentative.as_secs_f64() * 1e3);
        let (jc_mean, _) = us(|s| s.joint_calls as f64);
        let (tjc_mean, tjc_max) = us(|s| s.tentative_joint_calls as f64);
        eprintln!(
            "[{name}] steps={} ms/step mean (max): encoder {enc_mean:.1} ({enc_max:.1}) \
             decode {dec_mean:.2} ({dec_max:.2}) signals {sig_mean:.3} ({sig_max:.3}) \
             tentative {ten_mean:.2} ({ten_max:.2}); joint calls/step {jc_mean:.1}, \
             tentative {tjc_mean:.1} (max {tjc_max})",
            stats.len()
        );
        let out = serde_json::json!({
            "clip": name,
            "text": result.text,
            "emitted": emitted,
            "tokens": tokens,
            "committed": committed,
            "token_details": details,
            "frame_signals": frames,
            "steps": steps,
        });
        println!("{out}");
    }
    Ok(())
}
