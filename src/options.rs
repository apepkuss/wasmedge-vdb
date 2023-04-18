pub struct CreateCollectionOptions {
    shard_num: i32,
    consistency_level: ConsistencyLevel,
}
impl From<CreateCollectionOptions> for milvus::options::CreateCollectionOptions {
    fn from(options: CreateCollectionOptions) -> Self {
        milvus::options::CreateCollectionOptions::default()
            .shard_num(options.shard_num)
            .consistency_level(options.consistency_level.into())
    }
}

#[derive(Debug)]
pub enum ConsistencyLevel {
    Strong,
    Session,
    Bounded,
    Eventually,
    Customized,
}
impl From<ConsistencyLevel> for milvus::proto::common::ConsistencyLevel {
    fn from(level: ConsistencyLevel) -> Self {
        match level {
            ConsistencyLevel::Strong => milvus::proto::common::ConsistencyLevel::Strong,
            ConsistencyLevel::Session => milvus::proto::common::ConsistencyLevel::Session,
            ConsistencyLevel::Bounded => milvus::proto::common::ConsistencyLevel::Bounded,
            ConsistencyLevel::Eventually => milvus::proto::common::ConsistencyLevel::Eventually,
            ConsistencyLevel::Customized => milvus::proto::common::ConsistencyLevel::Customized,
        }
    }
}
