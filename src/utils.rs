use milvus::proto::common::{ErrorCode, MsgBase, MsgType, Status};

use crate::my_error::{Error, Result};

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
