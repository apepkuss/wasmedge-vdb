use crate::proto::{
    common::{ErrorCode, Status},
    schema::DataType,
};
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

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("try to set primary key {0:?}, but {1:?} is also key")]
    DuplicatePrimaryKey(String, String),

    #[error("can not find any primary key")]
    NoPrimaryKey,

    #[error("primary key must be int64 or varchar, unsupported type {0:?}")]
    UnsupportedPrimaryKey(DataType),

    #[error("auto id must be int64, unsupported type {0:?}")]
    UnsupportedAutoId(DataType),

    #[error("dimension mismatch for {0:?}, expected dim {1:?}, got {2:?}")]
    DimensionMismatch(String, i32, i32),

    #[error("wrong field data type, field {0} expected to be{1:?}, but got {2:?}")]
    FieldWrongType(String, DataType, DataType),

    #[error("field does not exists in schema: {0:?}")]
    FieldDoesNotExists(String),

    #[error("can not find such key {0:?}")]
    NoSuchKey(String),

    #[error("field {0:?} must be a vector field")]
    NotVectorField(String),
}

impl From<Status> for Error {
    fn from(s: Status) -> Self {
        Error::Server(ErrorCode::from_i32(s.error_code).unwrap(), s.reason)
    }
}

pub type Result<T> = result::Result<T, Error>;
