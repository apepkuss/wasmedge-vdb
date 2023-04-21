// use crate::collection::Error as CollectionError;
use crate::schema::SchemaError;
use milvus::proto::common::{ErrorCode, Status};
use std::result;
use thiserror::Error;
use tonic::transport::Error as CommError;
use tonic::Status as GrpcError;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0:?}")]
    Communication(#[from] CommError),

    // #[error("{0:?}")]
    // Collection(#[from] CollectionError),
    #[error("{0:?}")]
    Grpc(#[from] GrpcError),

    #[error("{0:?}")]
    Schema(#[from] SchemaError),

    #[error("{0:?} {1:?}")]
    Server(ErrorCode, String),

    #[error("{0:?}")]
    ProstEncode(#[from] prost::EncodeError),

    #[error("{0:?}")]
    ProstDecode(#[from] prost::DecodeError),

    #[error("Conversion error")]
    Conversion,
    #[error("{0:?}")]
    SerdeJsonErr(#[from] serde_json::Error),

    #[error("parameter {0:?} with invalid value {1:?}")]
    InvalidParameter(String, String),

    // #[error("{0:?}")]
    // Other(#[from] anyhow::Error),
    #[error("{0}")]
    Unexpected(String),
}

impl From<Status> for Error {
    fn from(s: Status) -> Self {
        Error::Server(ErrorCode::from_i32(s.error_code).unwrap(), s.reason)
    }
}

pub type Result<T> = result::Result<T, Error>;
