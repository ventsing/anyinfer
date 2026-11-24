//! KV 内存管理（Track E）。
//!
//! - [`arena`]（E-01）：物理块分配器，O(1) 队列 + 引用计数。
//! - [`prefix`]（E-02）：块粒度前缀树 + 祖先链引用计数。
//!
//! 这一层完全在 host 上跑，与设备无关 —— 除了最后把块表拷下去那一步（E-05）。
//! 所以这里能被完整测试，而上卡时只剩数据搬运。

pub mod arena;
pub mod prefix;

pub use arena::{Arena, ArenaConfig, ArenaError, Block};
pub use prefix::{CacheError, Evicted, Inserted, NodeId, PathRefs, PrefixCache, ROOT};
