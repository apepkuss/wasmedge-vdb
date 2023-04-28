use crate::error::{Error, Result};
use crate::{
    common::ConsistencyLevel,
    proto::common::{ErrorCode, MsgBase, MsgType, Status},
};

const STRONG_TIMESTAMP: u64 = 0;
const BOUNDED_TIMESTAMP: u64 = 2;
const EVENTUALLY_TIMESTAMP: u64 = 1;

pub fn new_msg(mtype: MsgType) -> MsgBase {
    MsgBase {
        msg_type: mtype as i32,
        timestamp: 0,
        source_id: 0,
        msg_id: 0,
        target_id: 0,
    }
}

pub fn status_to_result(status: &Option<Status>) -> Result<()> {
    let status = status
        .clone()
        .ok_or(Error::Unexpected("no status".to_owned()))?;

    match ErrorCode::from_i32(status.error_code) {
        Some(i) => match i {
            ErrorCode::Success => Ok(()),
            _ => Err(Error::from(status)),
        },
        None => Err(Error::Unexpected(format!(
            "unknown error code {}",
            status.error_code
        ))),
    }
}

pub async fn get_gts(level: ConsistencyLevel) -> u64 {
    match level {
        ConsistencyLevel::Strong => STRONG_TIMESTAMP,
        ConsistencyLevel::Bounded => BOUNDED_TIMESTAMP,
        ConsistencyLevel::Eventually => EVENTUALLY_TIMESTAMP,
        // TODO: NOT IMPLEMENTED
        ConsistencyLevel::Session => unimplemented!(),
        // TODO: THIS LEVEL DOES NOT WORK FOR NOW
        ConsistencyLevel::Customized => 0,
    }
}
