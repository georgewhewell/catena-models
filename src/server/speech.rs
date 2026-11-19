//! Phonon speech synthesis through the shared GPU worker.
use super::{
    api::{self, ServerState},
    engine::{Failure, Job},
};
use axum::{
    Json,
    extract::{
        Path, State,
        rejection::{JsonRejection, PathRejection},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use catena_models::phonon::{DEFAULT_MODEL, Phonon, UnknownVoice};
use serde::Deserialize;
use tokio::sync::oneshot;

pub(super) const MODELS: &[&str] = &[
    DEFAULT_MODEL,
    "gradium/phonon-40m-multilingual",
    "gradium/phonon-40m-en",
    "gradium/phonon-40m-fr",
    "gradium/phonon-40m-de",
    "gradium/phonon-40m-pt",
    "gradium/phonon-40m-es",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConvertRequest {
    text: String,
    model_id: String,
}

pub(super) async fn convert(
    State(state): State<ServerState>,
    headers: HeaderMap,
    path: Result<Path<String>, PathRejection>,
    body: Result<Json<ConvertRequest>, JsonRejection>,
) -> Response {
    if let Err(status) = api::authorize(&state, &headers) {
        return api::error(
            status,
            "authentication_error",
            "invalid or missing bearer token",
        );
    }
    let Path(voice) = match path {
        Ok(path) => path,
        Err(err) => return api::failure(Failure::invalid(err.body_text())),
    };
    let Json(request) = match body {
        Ok(body) => body,
        Err(err) => return api::failure(Failure::invalid(err.body_text())),
    };
    let voice = voice.trim().to_owned();
    if voice.is_empty() || request.text.trim().is_empty() {
        return api::failure(Failure::invalid("text and voice must not be blank"));
    }
    if !MODELS.contains(&request.model_id.as_str()) {
        return api::failure(Failure::invalid("unknown Phonon model_id"));
    }
    let (output, rx) = oneshot::channel();
    let job = Job::Synthesize {
        text: request.text,
        model_id: request.model_id,
        voice,
        output,
    };
    if let Err(response) = api::try_enqueue(&state, job) {
        return *response;
    }
    match rx.await {
        Ok(Ok(wav)) => wav.into_response(),
        Ok(Err(error)) => api::failure(error),
        Err(_) => api::error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "inference worker is unavailable",
        ),
    }
}

#[derive(Default)]
pub(super) struct VoiceCache {
    active: Option<(String, String, Phonon)>,
}

impl VoiceCache {
    pub(super) fn synthesize(
        &mut self,
        model_id: String,
        voice: String,
        text: &str,
    ) -> Result<String, Failure> {
        if self
            .active
            .as_ref()
            .is_none_or(|(model, name, _)| model != &model_id || name != &voice)
        {
            // Release the previous model's device buffers before loading its replacement.
            self.active = None;
            let phonon = Phonon::load_model_with_voice(&model_id, &voice).map_err(|error| {
                if error.is::<UnknownVoice>() {
                    Failure::invalid(error.to_string())
                } else {
                    speech_failure(error)
                }
            })?;
            self.active = Some((model_id, voice, phonon));
        }
        let (_, _, phonon) = self.active.as_ref().expect("Phonon loaded");
        let audio = phonon.synthesize(text, 0, false).map_err(speech_failure)?;
        Ok(STANDARD.encode(audio.wav_bytes().map_err(speech_failure)?))
    }
}

fn speech_failure(error: anyhow::Error) -> Failure {
    tracing::error!(error = %format_args!("{error:#}"), "Phonon synthesis failed");
    Failure::internal("Phonon synthesis failed")
}
