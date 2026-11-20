use anyhow::{Result, anyhow, ensure};
use tokenizers::Tokenizer;
use unicode_categories::UnicodeCategories;

const NONVERBAL: &[&str] = &[
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
];

// Inclusive Unicode block ends and phonetic weights from the OmniVoice
// RuleDurationEstimator (Xiaomi Corp., Apache-2.0). Adjacent blocks with
// the same weight are merged.
const SCRIPT_WEIGHTS: &[(u32, f64)] = &[
    (0x058F, 1.0),
    (0x08FF, 1.5),
    (0x0DFF, 1.8),
    (0x0EFF, 1.5),
    (0x109F, 1.8),
    (0x10FF, 1.0),
    (0x11FF, 2.5),
    (0x139F, 3.0),
    (0x177F, 1.0),
    (0x17FF, 1.8),
    (0x18FF, 1.0),
    (0x1C7F, 1.8),
    (0x1CBF, 1.0),
    (0x1CFF, 1.8),
    (0x1EFF, 1.0),
    (0x30FF, 2.2),
    (0x312F, 3.0),
    (0x318F, 2.5),
    (0xA4CF, 3.0),
    (0xA7FF, 1.0),
    (0xA82F, 1.8),
    (0xA87F, 1.0),
    (0xA95F, 1.8),
    (0xA97F, 2.5),
    (0xAAFF, 1.8),
    (0xAB2F, 3.0),
    (0xABBF, 1.0),
    (0xABFF, 1.8),
    (0xD7AF, 2.5),
    (0xFAFF, 3.0),
    (0xFDFF, 1.5),
    (0xFE6F, 1.0),
    (0xFEFF, 1.5),
    (0xFFEF, 1.0),
];

pub(super) fn duration_frames(text: &str) -> usize {
    let speed = weight("Nice to meet you.") / 25.0;
    let estimate = weight(text) / speed;
    let boosted = if estimate < 50.0 {
        50.0 * (estimate / 50.0).powf(1.0 / 3.0)
    } else {
        estimate
    };
    (boosted as usize).max(1)
}

fn weight(text: &str) -> f64 {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphabetic() {
                return 1.0;
            }
            if ch == ' ' || ch.is_separator() {
                return 0.2;
            }
            if ch == '\u{0640}' || ch.is_mark() {
                return 0.0;
            }
            if ch.is_punctuation() || ch.is_symbol() {
                return 0.5;
            }
            if ch.is_number() {
                return 3.5;
            }
            let code = ch as u32;
            let index = SCRIPT_WEIGHTS.partition_point(|&(end, _)| end < code);
            SCRIPT_WEIGHTS
                .get(index)
                .map_or(if code > 0x20000 { 3.0 } else { 1.0 }, |&(_, weight)| {
                    weight
                })
        })
        .sum()
}

fn format_text(text: &str) -> String {
    let mut replaced = String::with_capacity(text.len());
    let mut space = false;
    for ch in text.trim().chars() {
        match ch {
            '\r' | '\n' => continue,
            ' ' | '\t' if space => continue,
            _ => {}
        }
        space = matches!(ch, ' ' | '\t');
        replaced.push(match ch {
            '（' => '(',
            '）' => ')',
            ' ' | '\t' => ' ',
            _ => ch,
        });
    }
    let is_han = |ch| ('\u{4E00}'..='\u{9FFF}').contains(&ch);
    let mut formatted = String::new();
    let mut chars = replaced.chars().peekable();
    let mut previous = None;
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            let mut spaces = String::from(ch);
            while chars.peek().is_some_and(|ch| ch.is_whitespace()) {
                spaces.push(chars.next().unwrap());
            }
            if !previous.is_some_and(is_han) && !chars.peek().copied().is_some_and(is_han) {
                formatted.push_str(&spaces);
            }
        } else {
            formatted.push(ch);
            previous = Some(ch);
        }
    }
    formatted
}

pub(super) fn tokens(
    tokenizer: &Tokenizer,
    text: &str,
    instruct: Option<&str>,
) -> Result<Vec<u64>> {
    let text = format_text(text);
    ensure!(!text.trim().is_empty(), "text must contain a spoken prompt");
    let instruct = instruct.map(str::trim).unwrap_or("None");
    ensure!(
        !instruct.is_empty(),
        "OmniVoice instruction must not be empty"
    );
    ensure!(
        !instruct.contains("<|"),
        "OmniVoice instruction contains a reserved control delimiter"
    );
    let style =
        format!("<|lang_start|>None<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>");
    let wrapped = format!("<|text_start|>{text}<|text_end|>");
    let mut result = Vec::new();
    let mut encode = |segment: &str, special| -> Result<()> {
        if !segment.is_empty() {
            let ids = tokenizer
                .encode(segment, special)
                .map_err(|err| anyhow!("{err}"))?;
            result.extend(ids.get_ids().iter().map(|&id| u64::from(id)));
        }
        Ok(())
    };
    encode(&style, true)?;
    let mut remaining = wrapped.as_str();
    let mut split = false;
    while let Some((index, tag)) = NONVERBAL
        .iter()
        .filter_map(|&tag| remaining.find(tag).map(|index| (index, tag)))
        .min_by_key(|&(index, _)| index)
    {
        encode(&remaining[..index], false)?;
        encode(tag, false)?;
        remaining = &remaining[index + tag.len()..];
        split = true;
    }
    encode(remaining, !split)?;
    Ok(result)
}
