//! LFM3 d1 System One decisions: one complete prompt per question, no generation.
use std::{collections::HashSet, fmt, path::Path, rc::Rc};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use crate::{
    catena::InferenceOptions,
    checkpoint::{read_gguf_model_config, read_metadata},
    model::Model,
    tokenizer::QwenTokenizer,
    vision::{ImageEmbedding, PreparedMultimodalPrompt, VisionModel, image::ImagePatches},
};

#[derive(Debug)]
pub struct InvalidRequest(pub String);
impl fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for InvalidRequest {}
fn invalid(error: impl fmt::Display) -> anyhow::Error {
    InvalidRequest(error.to_string()).into()
}

#[derive(Clone, Debug)]
pub struct Question {
    pub name: String,
    pub kind: String,
    pub instructions: Value,
    pub options: Vec<(String, Value)>,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub state: Value,
    pub questions: Vec<Question>,
    images: Vec<Vec<u8>>,
    model: Option<String>,
}
impl Request {
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
    pub fn parse(body: Value) -> Result<Self> {
        Self::parse_inner(body).map_err(|e| invalid(format!("{e:#}")))
    }
    fn parse_inner(mut body: Value) -> Result<Self> {
        ensure!(body.is_object(), "request must be an object");
        ensure!(body.get("state").is_some(), "state must be provided");
        let mut images = Vec::new();
        for key in ["files", "images"] {
            if let Some(files) = body.get(key).filter(|v| !v.is_null()) {
                for file in files.as_array().context("files/images must be arrays")? {
                    load_image(file, &mut images)?;
                }
            }
        }
        if let Some(videos) = body.get("videos") {
            ensure!(
                videos.is_null() || videos.as_array().is_some_and(Vec::is_empty),
                "video is unsupported"
            );
        }
        let state = &mut body["state"];
        let messages = if state.is_object() {
            state.get_mut("messages")
        } else {
            Some(&mut *state)
        };
        if let Some(Value::Array(messages)) = messages {
            for message in messages {
                if let Some(Value::Array(parts)) = message.get_mut("content") {
                    let mut kept = Vec::new();
                    for part in std::mem::take(parts) {
                        match part.get("type").and_then(Value::as_str) {
                            Some("image_url") => {
                                let url = part.get("image_url").context("missing image_url")?;
                                load_image(url.get("url").unwrap_or(url), &mut images)?;
                            }
                            Some("input_audio" | "video_url") => {
                                bail!("audio/video is unsupported")
                            }
                            _ => kept.push(part),
                        }
                    }
                    *parts = kept;
                }
            }
        }
        let questions = body["questions"]
            .as_object()
            .context("questions must be a non-empty object")?;
        ensure!(
            !questions.is_empty(),
            "questions must be a non-empty object"
        );
        let mut parsed = Vec::new();
        for (name, q) in questions {
            parsed.push(parse_question(name, q).with_context(|| format!("question {name:?}"))?);
        }
        // Never silently accept generation or Python-only readout switches.
        for (key, value) in body.as_object().unwrap() {
            ensure!(
                matches!(
                    key.as_str(),
                    "state" | "questions" | "files" | "images" | "videos" | "model"
                ) || value.is_null(),
                "unsupported request parameter: {key}"
            );
        }
        let model = body
            .get("model")
            .map(|v| {
                v.as_str()
                    .context("model must be a string")
                    .map(str::to_owned)
            })
            .transpose()?;
        Ok(Self {
            state: body["state"].take(),
            questions: parsed,
            images,
            model,
        })
    }
}
fn parse_question(name: &str, q: &Value) -> Result<Question> {
    ensure!(q.is_object(), "must be an object");
    let kind = q["type"]
        .as_str()
        .context("type must be noul, choice or score")?;
    let instructions = q
        .get("instructions")
        .filter(|v| !v.is_null())
        .context("instructions must be provided")?;
    ensure!(
        !matches!(
            instructions.get("type").and_then(Value::as_str),
            Some("audio" | "input_audio")
        ),
        "spoken instructions are unsupported"
    );
    ensure!(
        q.get("multiple").is_none_or(|v| v.is_null() || v == false),
        "multiple choice is unsupported"
    );
    for key in ["range", "step", "unit", "max"] {
        ensure!(
            q.get(key).is_none_or(Value::is_null),
            "unsupported question parameter: {key}"
        );
    }
    let criteria = &q["criteria"];
    let options = match kind {
        "noul" => {
            ensure!(
                criteria.is_null() || criteria.is_object(),
                "criteria must be an object"
            );
            vec![
                ("true".into(), criteria["true"].clone()),
                ("false".into(), criteria["false"].clone()),
            ]
        }
        "choice" => {
            let options = criteria
                .as_object()
                .context("criteria must be a non-empty object")?;
            ensure!(
                !options.is_empty() && options.len() <= 255,
                "choice requires 1 to 255 options"
            );
            options
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        }
        "score" => {
            let levels = criteria
                .as_array()
                .context("criteria must be an array of 2 to 10 levels")?;
            ensure!(
                (2..=10).contains(&levels.len()),
                "score requires 2 to 10 levels"
            );
            levels
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect()
        }
        _ => bail!("type must be noul, choice or score"),
    };
    Ok(Question {
        name: name.into(),
        kind: kind.into(),
        instructions: instructions.clone(),
        options,
    })
}
fn load_image(url: &Value, images: &mut Vec<Vec<u8>>) -> Result<()> {
    ensure!(images.len() < 8, "at most eight images are supported");
    let url = url.as_str().context("images must be base64 data URLs")?;
    let (header, payload) = url.split_once(',').context("invalid image data URL")?;
    ensure!(
        matches!(
            header,
            "data:image/png;base64" | "data:image/jpeg;base64" | "data:image/jpg;base64"
        ),
        "only PNG/JPEG base64 data URLs are supported"
    );
    let bytes = STANDARD.decode(payload).context("invalid image base64")?;
    // Validate bytes before dispatching a GPU job.
    crate::vision::image::decode_bytes(&bytes)?;
    images.push(bytes);
    Ok(())
}

pub struct Metadata {
    pub template: String,
    temperatures: Map<String, Value>,
}
impl Metadata {
    /// None means a normal generative model, not a decision model.
    pub fn read(path: &Path) -> Result<Option<Self>> {
        let metadata = read_metadata(path)?;
        let Some(kind) = metadata.get("lfm3.decision.type") else {
            return Ok(None);
        };
        ensure!(kind == "lfm3-d1", "unsupported decision model type: {kind}");
        ensure!(
            metadata.get("lfm3.attention.causal") == Some(&Value::Bool(false)),
            "d1 requires noncausal full attention"
        );
        let template = metadata
            .get("tokenizer.chat_template.systemone")
            .and_then(Value::as_str)
            .context("d1 has no systemone template")?
            .to_owned();
        let mut temperatures = Map::new();
        for (key, value) in metadata {
            if let Some(name) = key.strip_prefix("lfm3.decision.temperature.") {
                let t = value.as_f64().context("invalid decision temperature")?;
                ensure!(
                    t.is_finite() && t > 0.,
                    "decision temperature must be positive and finite"
                );
                temperatures.insert(name.into(), value);
            }
        }
        for kind in ["noul", "choice", "score"] {
            ensure!(
                temperatures.contains_key(kind),
                "missing decision temperature for {kind}"
            );
        }
        Ok(Some(Self {
            template,
            temperatures,
        }))
    }
    fn temperature(&self, q: &Question, media: bool) -> f64 {
        let bucket = match q.options.len() {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        let specific = format!("{}.{bucket}", q.kind);
        (if media {
            self.temperatures.get("image")
        } else {
            None
        })
        .or_else(|| self.temperatures.get(&specific))
        .or_else(|| self.temperatures.get(&q.kind))
        .and_then(Value::as_f64)
        .unwrap_or(1.)
    }
}

/// One complete text-only decision prompt with its readout parameters.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub tokens: Vec<u64>,
    /// Single-token forms of each option, in option order. Logits are max-pooled per group.
    pub groups: Vec<Vec<u64>>,
    pub temperature: f64,
}

/// CPU-only request preparation and logits readout. `Runner` executes the
/// language model between the two; other callers may execute it elsewhere.
///
/// Text prompts need only the entries of
/// `catena::generated_model_source_for_options`, called once per prepared
/// prompt. d1 supports both prefill options; the CLI and server default to
/// matrix cores (`with_wmma_prefill(true)`).
///
/// 1. `qwen3.init(capacity: u64)` returns the session state `{cache, history}`.
///    The capacity is the prompt's token count.
/// 2. `qwen3(tokens: u64[], cache, history, capacity: u64, start: u64 = 0,
///    ...weights in spec::weight_specs order)` returns
///    `{cache, history, logits, token}`. `logits` holds `vocabulary_size` BF16
///    values, which `respond` takes widened to F32, and `token` is one u64.
pub struct Decisions {
    tokenizer: QwenTokenizer,
    metadata: Metadata,
    environment: minijinja::Environment<'static>,
    controls: Vec<String>,
    limit: usize,
    vocabulary_size: usize,
}
impl Decisions {
    pub fn read(path: &Path, limit: usize) -> Result<Self> {
        let metadata = Metadata::read(path)?.context("this model does not support decisions")?;
        let tokenizer = QwenTokenizer::from_gguf(path)?.context("missing tokenizer")?;
        let mut environment = template_environment();
        environment.add_template_owned("systemone", metadata.template.clone())?;
        let mut controls = tokenizer.control_tokens();
        controls.retain(|s| s.len() > 1);
        controls.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let config = read_gguf_model_config(path)?;
        let limit = limit.min(config.max_position_embeddings);
        ensure!(limit > 0, "context size must be positive");
        Ok(Self {
            tokenizer,
            metadata,
            environment,
            controls,
            limit,
            vocabulary_size: config.vocabulary_size,
        })
    }
    /// One prompt per question, in request order. Images need `Runner`.
    pub fn prepare(&self, request: &Request) -> Result<Vec<Prepared>> {
        if request.has_images() {
            return Err(invalid("images are unsupported without an image projector"));
        }
        Ok(request
            .questions
            .iter()
            .zip(self.prompts(request, &[])?)
            .map(|(q, (prompt, groups))| Prepared {
                tokens: prompt.tokens,
                groups,
                temperature: self.metadata.temperature(q, false),
            })
            .collect())
    }
    /// `logits[i]` holds the vocabulary logits the model returned for `prepared[i]`.
    pub fn respond(
        &self,
        model: &str,
        request: &Request,
        prepared: &[Prepared],
        logits: &[Vec<f32>],
    ) -> Result<Value> {
        ensure!(
            prepared.len() == request.questions.len() && logits.len() == prepared.len(),
            "decision count mismatch"
        );
        let mut answers = Map::new();
        let mut input_tokens = 0usize;
        for ((q, prepared), logits) in request.questions.iter().zip(prepared).zip(logits) {
            crate::model::check_logits(logits, self.vocabulary_size)?;
            input_tokens = input_tokens
                .checked_add(prepared.tokens.len())
                .context("usage overflow")?;
            answers.insert(
                q.name.clone(),
                readout(q, &prepared.groups, logits, prepared.temperature)?,
            );
        }
        Ok(envelope(model, answers, input_tokens))
    }
    // Validate every expanded prompt before running the language model.
    fn prompts(
        &self,
        request: &Request,
        images: &[Rc<ImageEmbedding>],
    ) -> Result<Vec<(PreparedMultimodalPrompt, Vec<Vec<u64>>)>> {
        request
            .questions
            .iter()
            .map(|q| {
                let (rendered, groups) = render(
                    &self.environment,
                    &self.tokenizer,
                    &self.controls,
                    &request.state,
                    q,
                    images.len(),
                )?;
                let prompt =
                    PreparedMultimodalPrompt::from_rendered(&self.tokenizer, &rendered, images)?;
                ensure!(
                    !prompt.tokens().is_empty() && prompt.tokens().len() <= self.limit,
                    "expanded decision prompt exceeds context limit {}",
                    self.limit
                );
                Ok((prompt, groups))
            })
            .collect::<Result<Vec<_>>>()
            .map_err(invalid)
    }
}
fn readout(q: &Question, groups: &[Vec<u64>], logits: &[f32], temperature: f64) -> Result<Value> {
    let scores = groups
        .iter()
        .map(|g| {
            g.iter()
                .map(|&id| logits[id as usize] as f64)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect::<Vec<_>>();
    answer(q, &scores, temperature)
}
fn envelope(model: &str, answers: Map<String, Value>, input_tokens: usize) -> Value {
    json!({"model":model,"answers":answers,"usage":{"input_tokens":input_tokens,"output_tokens":0}})
}

/// Loaded once on the owning GPU thread; image encodings live for one request.
pub struct Runner {
    model: Model,
    vision: Option<VisionModel>,
    decisions: Decisions,
    name: String,
}
impl Runner {
    pub fn load(
        path: &Path,
        projector: Option<&Path>,
        options: InferenceOptions,
        limit: usize,
        name: String,
    ) -> Result<Self> {
        let decisions = Decisions::read(path, limit)?;
        let model = if projector.is_some() {
            Model::load_gguf_with_multimodal_options(path, options)?
        } else {
            Model::load_gguf_with_options(path, options)?
        };
        let vision = projector
            .map(|path| VisionModel::load_gguf(path, model.gpu_dialect(), None))
            .transpose()?;
        if let Some(vision) = &vision {
            ensure!(
                vision.config().kind == crate::vision::VisionKind::Lfm3
                    && vision.config().output_size == model.config().hidden_size,
                "projector does not match LFM3 model"
            );
        }
        Ok(Self {
            model,
            vision,
            decisions,
            name,
        })
    }
    pub fn run(&self, request: &Request) -> Result<Value> {
        self.run_cancellable(request, || false)
    }
    pub fn run_cancellable(
        &self,
        request: &Request,
        cancelled: impl Fn() -> bool,
    ) -> Result<Value> {
        if request.model.as_ref().is_some_and(|m| m != &self.name) {
            return Err(invalid("unknown model"));
        }
        ensure!(
            request.images.is_empty() || self.vision.is_some(),
            "this server has no image projector"
        );
        let mut images: Vec<Rc<ImageEmbedding>> = Vec::new();
        if let Some(vision) = &self.vision {
            for bytes in &request.images {
                ensure!(!cancelled(), "request cancelled");
                let (rgb, w, h) = crate::vision::image::decode_bytes(bytes).map_err(invalid)?;
                let patches =
                    ImagePatches::from_rgb(&rgb, w, h, vision.config()).map_err(invalid)?;
                images.push(Rc::new(vision.encode_patches(&patches)?));
            }
        }
        let prepared = self.decisions.prompts(request, &images)?;
        let mut answers = Map::new();
        let mut input_tokens = 0usize;
        for (q, (prompt, groups)) in request.questions.iter().zip(prepared) {
            ensure!(!cancelled(), "request cancelled");
            input_tokens = input_tokens
                .checked_add(prompt.tokens().len())
                .context("usage overflow")?;
            let mut session = self.model.session(prompt.tokens().len())?;
            let logits = if images.is_empty() {
                session.prefill_logits(prompt.tokens())?
            } else {
                session.prefill_multimodal_logits(&prompt)?
            };
            let temperature = self.decisions.metadata.temperature(q, !images.is_empty());
            answers.insert(q.name.clone(), readout(q, &groups, &logits, temperature)?);
        }
        Ok(envelope(&self.name, answers, input_tokens))
    }
}
// llama.cpp's tojson uses Python separators and does not HTML-escape model input.
fn template_environment() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter(
        "tojson",
        |value: minijinja::Value,
         kwargs: minijinja::value::Kwargs|
         -> std::result::Result<minijinja::Value, minijinja::Error> {
            let indent: Option<usize> = kwargs.get("indent")?;
            kwargs.assert_all_used()?;
            let value = serde_json::to_value(value).map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })?;
            let text = if let Some(indent) = indent {
                use serde::Serialize;
                let mut bytes = Vec::new();
                let spaces = vec![b' '; indent];
                let formatter = serde_json::ser::PrettyFormatter::with_indent(&spaces);
                value
                    .serialize(&mut serde_json::Serializer::with_formatter(
                        &mut bytes, formatter,
                    ))
                    .unwrap();
                String::from_utf8(bytes).unwrap()
            } else {
                json_text(&value)
            };
            Ok(minijinja::Value::from_safe_string(text))
        },
    );
    env
}
fn json_text(value: &Value) -> String {
    match value {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(json_text).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), json_text(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => value.to_string(),
    }
}

fn escape(value: &Value, controls: &[String]) -> Value {
    match value {
        Value::String(s) => Value::String(break_tokens(s, controls)),
        Value::Array(a) => Value::Array(a.iter().map(|v| escape(v, controls)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (break_tokens(k, controls), escape(v, controls)))
                .collect(),
        ),
        _ => value.clone(),
    }
}
fn break_tokens(text: &str, controls: &[String]) -> String {
    let mut out = text.to_owned();
    loop {
        let mut changed = false;
        for token in controls {
            if out.contains(token) {
                let boundary = token.chars().next().unwrap().len_utf8();
                out = out.replace(
                    token,
                    &format!("{}\u{200b}{}", &token[..boundary], &token[boundary..]),
                );
                changed = true;
            }
        }
        if !changed {
            return out;
        }
    }
}
fn render(
    environment: &minijinja::Environment<'_>,
    tokenizer: &QwenTokenizer,
    controls: &[String],
    state: &Value,
    q: &Question,
    images: usize,
) -> Result<(String, Vec<Vec<u64>>)> {
    let (labels, groups) = label_groups(tokenizer, q)?;
    let options = q.options.iter().zip(labels).map(|((key,description),label)| json!({"key":key,"description":description,"label":label})).collect::<Vec<_>>();
    let input = escape(
        &json!({"state":state,"instructions":q.instructions,"type":q.kind,"options":options}),
        controls,
    );
    let rendered = environment.get_template("systemone")?.render(minijinja::context! {
        state => input["state"], instructions => input["instructions"], type => input["type"], options => input["options"],
        images => vec!["<|vision_start|><|image|><|vision_end|>";images],
    })?;
    Ok((rendered, groups))
}
fn singles(tokenizer: &QwenTokenizer, forms: &[String]) -> Result<Vec<u64>> {
    let mut ids = Vec::new();
    for form in forms {
        let tokens = tokenizer.encode_without_special_tokens(form)?;
        if tokens.len() == 1 && !ids.contains(&tokens[0]) {
            ids.push(tokens[0]);
        }
    }
    Ok(ids)
}
fn label_groups(tokenizer: &QwenTokenizer, q: &Question) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    let mut labels = Vec::new();
    let mut groups = Vec::new();
    if q.kind != "choice" {
        for (key, _) in &q.options {
            let forms: Vec<String> = match key.as_str() {
                "true" => ["yes", "Yes", "YES"].map(str::to_owned).to_vec(),
                "false" => ["no", "No", "NO"].map(str::to_owned).to_vec(),
                _ => vec![key.clone()],
            };
            let group = singles(tokenizer, &forms)?;
            ensure!(!group.is_empty(), "no single-token label for {key}");
            labels.push(key.clone());
            groups.push(group);
        }
        return Ok((labels, groups));
    }
    let letters = q
        .options
        .iter()
        .all(|(k, _)| k.len() == 1 && k.as_bytes()[0].is_ascii_alphabetic());
    let mut pool = ('A'..='Z').map(|c| c.to_string()).collect::<Vec<_>>();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend(('a'..='z').map(|c| c.to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    pool.extend(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))));
    let mut used = HashSet::new();
    for (i, (key, _)) in q.options.iter().enumerate() {
        let code = if letters {
            key.clone()
        } else if i < 26 {
            ((b'A' + i as u8) as char).to_string()
        } else {
            format!(
                "{}{}",
                (b'A' + ((i - 26) / 26) as u8) as char,
                (b'A' + ((i - 26) % 26) as u8) as char
            )
        };
        let mut found = None;
        for candidate in std::iter::once(&code).chain(pool.iter()) {
            let ids = tokenizer.encode_without_special_tokens(candidate)?;
            if ids.len() == 1 && used.insert(ids[0]) {
                found = Some((candidate.clone(), ids[0]));
                break;
            }
        }
        let (code, id) = found.context("no distinct single-token alias available")?;
        let mut group = vec![id];
        for id in singles(tokenizer, &[format!(" {code}")])? {
            if !group.contains(&id) {
                group.push(id);
            }
        }
        labels.push(code);
        groups.push(group);
    }
    Ok((labels, groups))
}
fn answer(q: &Question, scores: &[f64], temperature: f64) -> Result<Value> {
    ensure!(
        scores.iter().all(|s| s.is_finite()),
        "nonfinite decision logits"
    );
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs = scores
        .iter()
        .map(|s| ((s - max) / temperature).exp())
        .collect::<Vec<_>>();
    let total: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }
    let mut best = 0;
    for i in 1..probs.len() {
        if probs[i] > probs[best] {
            best = i;
        }
    }
    if q.kind == "noul" {
        return Ok(json!({"type":"noul","noul":probs[0]}));
    }
    let probabilities: Map<String, Value> = q
        .options
        .iter()
        .zip(&probs)
        .map(|((k, _), p)| (k.clone(), json!(p)))
        .collect();
    if q.kind == "choice" {
        return Ok(
            json!({"type":"choice","choice":q.options[best].0,"confidence":probs[best],"probabilities":probabilities}),
        );
    }
    let mut mass = 0.;
    let median = probs
        .iter()
        .position(|p| {
            mass += p;
            mass >= 0.5
        })
        .unwrap_or(probs.len() - 1);
    let legend: Map<String, Value> = q.options.iter().cloned().collect();
    Ok(
        json!({"type":"score","score":median as f64,"confidence":probs[best],"probabilities":probabilities,"legend":legend}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Value {
        json!({"state":"I was charged twice.","questions":{
            "refund":{"type":"noul","instructions":"Refund requested?"},
            "team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","support":null}},
            "urgency":{"type":"score","instructions":"How urgent?","criteria":["Can wait","Today","Now"]}
        }})
    }
    #[test]
    fn requests_and_readout() -> Result<()> {
        let request = Request::parse(sample())?;
        assert_eq!(
            request
                .questions
                .iter()
                .map(|q| q.name.as_str())
                .collect::<Vec<_>>(),
            ["refund", "team", "urgency"]
        );
        let binary = answer(&request.questions[0], &[2., 0.], 2.)?;
        assert!((binary["noul"].as_f64().unwrap() - 1. / (1. + (-1f64).exp())).abs() < 1e-12);
        let choice = answer(&request.questions[1], &[0., 2.], 1.)?;
        assert_eq!(choice["choice"], "support");
        assert_eq!(choice["confidence"], choice["probabilities"]["support"]);
        // Median 1, mode 2, and mean 1.28: catches accidental generic score readout.
        let score = answer(
            &request.questions[2],
            &[0.21f64.ln(), 0.3f64.ln(), 0.49f64.ln()],
            1.,
        )?;
        assert_eq!(score["score"], 1.0);
        assert_eq!(score["legend"]["1"], "Today");
        assert!(answer(&request.questions[0], &[f64::NAN, 0.], 1.).is_err());
        for bad in [
            json!({"questions":{}}),
            json!({"state":null,"questions":{}}),
            json!({"state":"x","questions":{"x":{"type":"number","instructions":"x"}}}),
            json!({"state":"x","questions":{"x":{"type":"choice","instructions":"x","criteria":{"a":null},"multiple":true}}}),
        ] {
            assert!(Request::parse(bad).unwrap_err().is::<InvalidRequest>());
        }
        Ok(())
    }
    fn image_request() -> Result<Request> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.write_header()?.write_image_data(&[255, 0, 0])?;
        }
        let image = format!("data:image/png;base64,{}", STANDARD.encode(bytes));
        let mut request = sample();
        request["state"] = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":image}},{"type":"text","text":"hello"}]}]});
        Request::parse(request)
    }
    #[test]
    fn media_validation_and_escaping() -> Result<()> {
        let controls = vec!["<|im_end|>".into(), "<|image|>".into()];
        assert_eq!(
            break_tokens("<|im_end|><|image|>", &controls),
            "<\u{200b}|im_end|><\u{200b}|image|>"
        );
        let mut request = sample();
        request["images"] = json!(["https://example.com/photo.png"]);
        assert!(Request::parse(request).is_err());
        let request = image_request()?;
        assert!(request.has_images());
        assert_eq!(request.model(), None);
        assert_eq!(
            request.state["messages"][0]["content"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        Ok(())
    }
    #[test]
    fn template_json_matches_llama_separators() -> Result<()> {
        let mut env = template_environment();
        env.add_template(
            "json",
            "{{ value | tojson }}\n{{ value | tojson(indent=2) }}",
        )?;
        let rendered = env
            .get_template("json")?
            .render(minijinja::context! {value => json!({"x":"<a>& café","a":[true,null]})})?;
        assert_eq!(
            rendered,
            "{\"x\": \"<a>& café\", \"a\": [true, null]}\n{\n  \"x\": \"<a>& café\",\n  \"a\": [\n    true,\n    null\n  ]\n}"
        );
        Ok(())
    }
    pub(super) fn cached(name: &str) -> Result<std::path::PathBuf> {
        let root = std::env::var_os("HF_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
                    .join(".cache/huggingface")
            });
        for entry in std::fs::read_dir(root.join("hub/models--LiquidAI--d1-2B-GGUF/snapshots"))? {
            let path = entry?.path().join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        bail!("missing cached {name}")
    }
    #[test]
    #[ignore = "requires cached d1 BF16 GGUF; no GPU needed"]
    fn cached_metadata_template_and_graphs() -> Result<()> {
        use crate::catena;
        use catena_lang::codegen::GpuDialect;
        let path = cached("d1-2B-BF16.gguf")?;
        let metadata = Metadata::read(&path)?.unwrap();
        assert_eq!(
            metadata.template,
            include_str!("../tests/fixtures/d1_systemone.jinja")
        );
        let tokenizer = QwenTokenizer::from_gguf(&path)?.unwrap();
        let mut environment = template_environment();
        environment.add_template_owned("systemone", metadata.template)?;
        let request = Request::parse(sample())?;
        let (prompt, groups) = render(
            &environment,
            &tokenizer,
            &tokenizer.control_tokens(),
            &request.state,
            &request.questions[0],
            0,
        )?;
        assert_eq!(
            prompt,
            "<|startoftext|><|im_start|>user\nI was charged twice.\n\n\nQUESTION:\nRefund requested?\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
        assert_eq!(groups.len(), 2);
        let decisions = Decisions::read(&path, 8192)?;
        let prepared = decisions.prepare(&request)?;
        assert_eq!(prepared.len(), 3);
        assert_eq!(
            prepared[0].tokens,
            tokenizer.encode_without_special_tokens(&prompt)?
        );
        assert_eq!(prepared[0].groups, groups);
        let vocabulary = read_gguf_model_config(&path)?.vocabulary_size;
        let mut logits = vec![vec![0f32; vocabulary]; 3];
        logits[0][groups[0][1] as usize] = 2.;
        let response = decisions.respond("d1", &request, &prepared, &logits)?;
        assert_eq!(response["model"], "d1");
        assert_eq!(
            response["usage"],
            json!({"input_tokens": prepared.iter().map(|p| p.tokens.len()).sum::<usize>(), "output_tokens": 0})
        );
        let expected = answer(&request.questions[0], &[2., 0.], prepared[0].temperature)?;
        assert_eq!(response["answers"]["refund"], expected);
        assert_eq!(response["answers"]["urgency"]["type"], "score");
        logits[1][0] = f32::NAN;
        assert!(
            decisions
                .respond("d1", &request, &prepared, &logits)
                .is_err()
        );
        logits[1].pop();
        assert!(
            decisions
                .respond("d1", &request, &prepared, &logits)
                .is_err()
        );
        let error = decisions.prepare(&image_request()?).unwrap_err();
        assert!(error.is::<InvalidRequest>());
        let mut tiny = Decisions::read(&path, 8192)?;
        tiny.limit = 4;
        assert!(tiny.prepare(&request).unwrap_err().is::<InvalidRequest>());
        let mut many = request.questions[1].clone();
        many.options = (0..40).map(|i| (format!("opt{i}"), Value::Null)).collect();
        let (labels, groups) = label_groups(&tokenizer, &many)?;
        assert_eq!(labels[0], "A");
        assert_eq!(labels[26], "AA");
        assert_eq!(
            groups.iter().map(|g| g[0]).collect::<HashSet<_>>().len(),
            40
        );
        let config = read_gguf_model_config(&path)?;
        assert!(!config.lfm3.as_ref().unwrap().causal_attention);
        crate::checkpoint::validate_gguf_tensors(&path, &config)?;
        // The default graph contains both native batch and scalar fallback attention.
        let source = catena::generated_multimodal_source(
            &config,
            InferenceOptions::default().with_wmma_prefill(true),
        )?;
        for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
            eprintln!("compiling d1 graph: {dialect:?}");
            catena::compile_sources(catena::runtime_sources(&source), dialect)?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires HIP/CUDA GPU and cached d1 BF16 model/projector"]
    fn decisions_match_reference() -> Result<()> {
        let path = cached("d1-2B-BF16.gguf")?;
        let projector = cached("mmproj-d1-2B-BF16.gguf")?;
        let runner = Runner::load(
            &path,
            Some(&projector),
            InferenceOptions::default().with_wmma_prefill(std::env::var_os("D1_SIMT").is_none()),
            8192,
            "d1".into(),
        )?;
        let mut requests = vec![
            sample(),
            json!({"state":{"ticket":"Please refund the duplicate charge.","plan":"pro"},"questions":sample()["questions"]}),
        ];
        let mut long = sample();
        long["state"] = json!(format!(
            "{} Please refund the duplicate charge.",
            "Ordinary account history. ".repeat(80)
        ));
        requests.push(long);
        let mut urls = Vec::new();
        for rgb in [[255u8, 0, 0], [0, 0, 255]] {
            let mut bytes = Vec::new();
            {
                let mut encoder = png::Encoder::new(&mut bytes, 256, 256);
                encoder.set_color(png::ColorType::Rgb);
                encoder
                    .write_header()?
                    .write_image_data(&rgb.repeat(256 * 256))?;
            }
            urls.push(format!("data:image/png;base64,{}", STANDARD.encode(bytes)));
        }
        requests.push(json!({"state":null,"images":urls,"questions":{"color":{"type":"choice","instructions":"What is the color of the first image?","criteria":{"red":"Red","blue":"Blue","green":"Green"}}}}));
        let mut outputs = Vec::new();
        for request in &requests {
            let response = runner.run(&Request::parse(request.clone())?)?;
            assert_eq!(response["usage"]["output_tokens"], 0);
            assert!(response["usage"]["input_tokens"].as_u64().unwrap() > 0);
            outputs.push(response);
        }
        assert_eq!(outputs[3]["answers"]["color"]["choice"], "red");
        if let Ok(directory) = std::env::var("D1_TEST_OUTPUT") {
            std::fs::create_dir_all(&directory)?;
            std::fs::write(
                format!("{directory}/requests.json"),
                serde_json::to_vec_pretty(&requests)?,
            )?;
            std::fs::write(
                format!("{directory}/responses.json"),
                serde_json::to_vec_pretty(&outputs)?,
            )?;
        }
        if let Ok(reference) = std::env::var("D1_REFERENCE") {
            let reference: Vec<Value> = serde_json::from_slice(&std::fs::read(reference)?)?;
            assert_eq!(reference.len(), outputs.len());
            for (actual, expected) in outputs.iter().zip(reference) {
                assert_eq!(actual["usage"], expected["usage"]);
                for (name, answer) in actual["answers"].as_object().unwrap() {
                    let expected = &expected["answers"][name];
                    if answer["type"] == "noul" {
                        assert!(
                            (answer["noul"].as_f64().unwrap() - expected["noul"].as_f64().unwrap())
                                .abs()
                                < 0.02
                        );
                    } else {
                        for (key, p) in answer["probabilities"].as_object().unwrap() {
                            assert!(
                                (p.as_f64().unwrap()
                                    - expected["probabilities"][key].as_f64().unwrap())
                                .abs()
                                    < 0.02,
                                "{name}/{key}: {answer} vs {expected}"
                            );
                        }
                        if answer["type"] == "choice" {
                            assert_eq!(answer["choice"], expected["choice"]);
                        } else {
                            assert_eq!(answer["score"].as_f64(), expected["score"].as_f64());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
