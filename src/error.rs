/// The WasmEdge result type.
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VDBError {
    #[error("{0}")]
    Communication(String),
    #[error("{0}")]
    Collection(String),
    #[error("{0}")]
    Grpc(String),
    #[error("{0}")]
    Schema(String),
    #[error("{0}")]
    Server(String),
}
impl From<milvus::error::Error> for VDBError {
    fn from(e: milvus::error::Error) -> Self {
        match e {
            milvus::error::Error::Communication(e) => VDBError::Communication(e.to_string()),
            milvus::error::Error::Collection(e) => VDBError::Collection(e.to_string()),
            milvus::error::Error::Grpc(e) => VDBError::Grpc(e.to_string()),
            milvus::error::Error::Schema(e) => VDBError::Schema(e.to_string()),
            milvus::error::Error::Server(code, message) => {
                VDBError::Server(format!("{:?}, {}", code, message))
            }
            _ => panic!("Unsupported error type! error: {e:?}"),
        }
    }
}

pub type VDBResult<T> = Result<T, Box<VDBError>>;
