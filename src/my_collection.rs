use crate::{common::ConsistencyLevel, schema::CollectionSchema};

#[derive(Debug, Clone)]
pub struct CollectionMetadata {
    pub name: String,
    pub id: i64,
    /// The collection schema
    pub schema: Option<CollectionSchema>,
    /// Hybrid timestamp in milvus
    pub created_timestamp: u64,
    /// The utc timestamp calculated by created_timestamp
    pub created_utc_timestamp: u64,
    /// The shards number
    pub shards_num: i32,
    /// The aliases of this collection
    pub aliases: Vec<String>,
    /// The consistency level that the collection used
    pub consistency_level: ConsistencyLevel,
}

#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub name: String,
    pub id: i64,
    pub created_timestamp: u64,
    pub created_utc_timestamp: u64,
    pub in_memory_percentage: i64,
    pub query_service_available: bool,
}

#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub name: String,
    pub id: i64,
    pub created_timestamp: u64,
    pub created_utc_timestamp: u64,
    pub in_memory_percentage: i64,
}
