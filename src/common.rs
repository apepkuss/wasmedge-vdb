use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum ConsistencyLevel {
    Strong = 0,
    /// default in PyMilvus
    Session = 1,
    Bounded = 2,
    Eventually = 3,
    /// Users pass their own `guarantee_timestamp`.
    Customized = 4,
}
impl From<ConsistencyLevel> for milvus::proto::common::ConsistencyLevel {
    fn from(level: ConsistencyLevel) -> Self {
        milvus::proto::common::ConsistencyLevel::from_i32(level.to_i32().unwrap()).unwrap()
    }
}
impl From<milvus::proto::common::ConsistencyLevel> for ConsistencyLevel {
    fn from(level: milvus::proto::common::ConsistencyLevel) -> Self {
        ConsistencyLevel::from_i32(level as i32).unwrap()
    }
}
