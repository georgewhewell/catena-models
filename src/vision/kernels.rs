//! Select Hex helpers for the shared vision composition.
//! This renders a graph; allocation, kernels, scheduling and releases execute in Hex.
use super::{VisionConfig, graph};

pub(super) fn source(config: &VisionConfig, names: &[String]) -> String {
    let graph = graph::source(config, names);
    [
        include_str!("../catena/memory.hex"),
        include_str!("../catena/numeric.hex"),
        include_str!("../catena/resources.hex"),
        include_str!("../catena/qwen35_vision.hex"),
        &graph,
    ]
    .join("\n")
}
