#[derive(Debug)]
pub struct CreateCollectionOptions {
    shard_num: i32,
    consistency_level: ConsistencyLevel,
}
impl Default for CreateCollectionOptions {
    fn default() -> Self {
        Self {
            shard_num: 2,
            consistency_level: ConsistencyLevel::default(),
        }
    }
}
impl CreateCollectionOptions {
    pub fn new(shard_num: i32, level: ConsistencyLevel) -> Self {
        let mut options = Self::default();
        options.shard_num = shard_num;
        options.consistency_level = level;

        options
    }

    pub fn shard_num(&self) -> i32 {
        self.shard_num
    }

    pub fn consistency_level(&self) -> &ConsistencyLevel {
        &self.consistency_level
    }
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
impl Default for ConsistencyLevel {
    fn default() -> Self {
        ConsistencyLevel::Session
    }
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
