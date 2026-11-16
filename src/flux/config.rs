use crate::spec::{ModelArchitecture, ModelConfig};
use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;
const FILES: &[&str] = &[
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];
pub fn snapshot(model: &str) -> Result<PathBuf> {
    let snapshot = crate::hfhub::resolve_model_files(model, FILES)?;
    let single = "transformer/diffusion_pytorch_model.safetensors";
    let index = "transformer/diffusion_pytorch_model.safetensors.index.json";
    if snapshot.join(index).is_file() {
        crate::hfhub::resolve_model_files(model, &[index])
    } else {
        crate::hfhub::resolve_model_files(model, &[single])
            .or_else(|_| crate::hfhub::resolve_model_files(model, &[index]))
    }
}
/// Select the checkpoint format present in a resolved snapshot.
pub fn transformer_checkpoint(snapshot: &std::path::Path) -> PathBuf {
    let single = snapshot.join("transformer/diffusion_pytorch_model.safetensors");
    let index = single.with_extension("safetensors.index.json");
    if index.is_file() { index } else { single }
}
pub use crate::hfhub::read_json as read;
#[derive(Deserialize, Debug)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}
impl TextConfig {
    pub fn qwen(&self) -> Result<ModelConfig> {
        let mut c = ModelConfig::qwen3_0_6b();
        c.architecture = ModelArchitecture::Qwen3Dense;
        c.hidden_size = self.hidden_size;
        c.intermediate_size = self.intermediate_size;
        c.num_layers = 27;
        c.num_attention_heads = self.num_attention_heads;
        c.num_key_value_heads = self.num_key_value_heads;
        c.head_dim = self.head_dim;
        c.rope_dimension_count = self.head_dim;
        c.vocabulary_size = self.vocab_size;
        c.rms_norm_eps = self.rms_norm_eps;
        c.rope_theta = self.rope_theta;
        c.max_position_embeddings = self.max_position_embeddings;
        Ok(c)
    }
}
#[derive(Deserialize, Debug)]
pub struct TransformerConfig {
    pub attention_head_dim: usize,
    pub axes_dims_rope: Vec<usize>,
    pub eps: f32,
    pub guidance_embeds: bool,
    pub in_channels: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: f32,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub rope_theta: f32,
    pub timestep_guidance_channels: usize,
}
impl TransformerConfig {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    pub fn intermediate_size(&self) -> usize {
        (self.hidden_size() as f32 * self.mlp_ratio) as usize
    }
}
#[derive(Deserialize, Debug)]
pub struct VaeConfig {
    pub block_out_channels: Vec<usize>,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    pub batch_norm_eps: f32,
    pub patch_size: Vec<usize>,
    pub force_upcast: bool,
    pub layers_per_block: usize,
}
