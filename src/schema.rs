use crate::my_error::{Error, Result};
use std::fmt;

use milvus::proto::schema::DataType;

#[derive(Debug, Clone)]
pub struct CollectionSchema {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) fields: Vec<FieldSchema>,
}

impl CollectionSchema {
    pub fn new(name: &str, fields: Vec<FieldSchema>, description: Option<&str>) -> Result<Self> {
        let mut has_primary = false;

        for f in fields.iter() {
            if f.is_primary() {
                has_primary = true;
                break;
            }
        }

        if !has_primary {
            return Err(Error::from(SchemaError::NoPrimaryKey));
        }

        // let this = std::mem::replace(self, CollectionSchemaBuilder::new("".into(), ""));

        Ok(CollectionSchema {
            fields: fields.into_iter().map(|x| x.into()).collect(),
            name: name.into(),
            description: description.unwrap_or_default().into(),
        })
    }

    // pub fn primary_column(&self) -> Option<&FieldSchema> {
    //     self.fields.iter().find(|s| s.is_primary)
    // }

    // pub fn validate(&self) -> Result<()> {
    //     self.primary_column()
    //         .ok_or_else(|| SchemaError::NoPrimaryKey)?;
    //     // TODO addidtional schema checks need to be added here
    //     Ok(())
    // }

    // pub fn get_field<S>(&self, name: S) -> Option<&FieldSchema>
    // where
    //     S: AsRef<str>,
    // {
    //     let name = name.as_ref();
    //     self.fields.iter().find(|f| f.name == name)
    // }

    // pub fn is_valid_vector_field(&self, field_name: &str) -> Result<()> {
    //     for f in &self.fields {
    //         if f.name == field_name {
    //             if f.dtype == DataType::BinaryVector || f.dtype == DataType::FloatVector {
    //                 return Ok(());
    //             } else {
    //                 return Err(Error::from(SchemaError::NotVectorField(
    //                     field_name.to_owned(),
    //                 )));
    //             }
    //         }
    //     }
    //     return Err(error::Error::from(SchemaError::NoSuchKey(
    //         field_name.to_owned(),
    //     )));
    // }
}
impl From<CollectionSchema> for milvus::proto::schema::CollectionSchema {
    fn from(schema: CollectionSchema) -> Self {
        Self {
            name: schema.name.to_string(),
            description: schema.description,
            auto_id: false,
            fields: schema.fields.into_iter().map(Into::into).collect(),
        }
    }
}
impl From<milvus::proto::schema::CollectionSchema> for CollectionSchema {
    fn from(schema: milvus::proto::schema::CollectionSchema) -> Self {
        CollectionSchema {
            name: schema.name,
            description: schema.description,
            fields: schema.fields.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Clone)]
pub struct FieldSchema {
    name: String,
    desc: String,
    ty: FieldType,
}
impl FieldSchema {
    pub fn new(name: &str, ty: FieldType, description: Option<&str>) -> Self {
        let desc = match description {
            Some(desc) => desc.to_string(),
            None => String::new(),
        };

        Self {
            name: name.to_string(),
            desc,
            ty,
        }
    }

    pub fn is_primary(&self) -> bool {
        match &self.ty {
            FieldType::Int64(pk, _) => *pk,
            FieldType::VarChar(_, pk, _) => *pk,
            _ => false,
        }
    }
}
impl fmt::Debug for FieldSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = match &self.ty {
            FieldType::None => format!("dtype: None"),
            FieldType::Bool => format!("dtype: Bool"),
            FieldType::Int8 => format!("dtype: Int8"),
            FieldType::Int16 => format!("dtype: Int16"),
            FieldType::Int32 => format!("dtype: Int32"),
            FieldType::Int64(pk, auto_id) => {
                format!("dtype: Int64, is_primary: {pk}, auto_id: {auto_id}")
            }
            FieldType::Float => format!("dtype: Float"),
            FieldType::Double => format!("dtype: Double"),
            FieldType::String => format!("dtype: String"),
            FieldType::VarChar(max_length, pk, auto_id) => {
                format!("dtype: Varchar, max_length: {max_length}, is_primary: {pk}, auto_id: {auto_id}")
            }
            FieldType::BinaryVector(dim) => format!("dtype: BinaryVector, dimension: {dim}"),
            FieldType::FloatVector(dim) => format!("dtype: FloatVector, dimension: {dim}"),
        };

        let message = format!(
            "name: {name}, description: {desc}, {ty}",
            name = self.name,
            desc = self.desc,
            ty = ty,
        );

        write!(f, "{}", message)
    }
}
// impl From<milvus::schema::FieldSchema> for FieldSchema {
//     fn from(field: milvus::schema::FieldSchema) -> Self {
//         let ty = match field.dtype {
//             milvus::proto::schema::DataType::None => FieldType::None,
//             milvus::proto::schema::DataType::Bool => FieldType::Bool,
//             milvus::proto::schema::DataType::Int8 => FieldType::Int8,
//             milvus::proto::schema::DataType::Int16 => FieldType::Int16,
//             milvus::proto::schema::DataType::Int32 => FieldType::Int32,
//             milvus::proto::schema::DataType::Int64 => match field.is_primary {
//                 true => FieldType::Int64(true, field.auto_id),
//                 false => FieldType::Int64(false, field.auto_id),
//             },
//             milvus::proto::schema::DataType::Float => FieldType::Float,
//             milvus::proto::schema::DataType::Double => FieldType::Double,
//             milvus::proto::schema::DataType::String => FieldType::String,
//             milvus::proto::schema::DataType::VarChar => match field.is_primary {
//                 true => FieldType::VarChar(field.max_length, true, field.auto_id),
//                 false => FieldType::VarChar(field.max_length, false, field.auto_id),
//             },
//             milvus::proto::schema::DataType::BinaryVector => FieldType::BinaryVector(field.dim),
//             milvus::proto::schema::DataType::FloatVector => FieldType::FloatVector(field.dim),
//         };

//         Self {
//             name: field.name,
//             desc: field.description,
//             ty,
//         }
//     }
// }
// impl From<FieldSchema> for milvus::schema::FieldSchema {
//     fn from(field: FieldSchema) -> Self {
//         match field.ty {
//             FieldType::None => milvus::schema::FieldSchema::default(),
//             FieldType::Bool => milvus::schema::FieldSchema::new_bool(&field.name, &field.desc),
//             FieldType::Int8 => milvus::schema::FieldSchema::new_int8(&field.name, &field.desc),
//             FieldType::Int16 => milvus::schema::FieldSchema::new_int16(&field.name, &field.desc),
//             FieldType::Int32 => milvus::schema::FieldSchema::new_int32(&field.name, &field.desc),
//             FieldType::Int64(pk, auto_id) => match pk {
//                 true => milvus::schema::FieldSchema::new_primary_int64(
//                     &field.name,
//                     &field.desc,
//                     auto_id,
//                 ),
//                 false => milvus::schema::FieldSchema::new_int64(&field.name, &field.desc),
//             },
//             FieldType::Float => milvus::schema::FieldSchema::new_float(&field.name, &field.desc),
//             FieldType::Double => milvus::schema::FieldSchema::new_double(&field.name, &field.desc),
//             FieldType::String => milvus::schema::FieldSchema::new_string(&field.name, &field.desc),
//             FieldType::VarChar(max_length, pk, auto_id) => match pk {
//                 true => milvus::schema::FieldSchema::new_primary_varchar(
//                     &field.name,
//                     &field.desc,
//                     auto_id,
//                     max_length,
//                 ),
//                 false => {
//                     milvus::schema::FieldSchema::new_varchar(&field.name, &field.desc, max_length)
//                 }
//             },
//             FieldType::BinaryVector(dimension) => {
//                 milvus::schema::FieldSchema::new_binary_vector(&field.name, &field.desc, dimension)
//             }
//             FieldType::FloatVector(dimension) => {
//                 milvus::schema::FieldSchema::new_float_vector(&field.name, &field.desc, dimension)
//             }
//         }
//     }
// }
impl From<FieldSchema> for milvus::proto::schema::FieldSchema {
    fn from(field: FieldSchema) -> Self {
        let mut is_primary_key = false;
        let mut auto_id = false;
        let type_params = match field.ty {
            FieldType::Int64(pk, auto) => {
                is_primary_key = pk;
                auto_id = auto;
                vec![]
            }
            FieldType::VarChar(max_length, pk, auto) => {
                is_primary_key = pk;
                auto_id = auto;
                vec![milvus::proto::common::KeyValuePair {
                    key: "max_length".to_string(),
                    value: max_length.to_string(),
                }]
            }
            FieldType::BinaryVector(dimension) | FieldType::FloatVector(dimension) => {
                vec![milvus::proto::common::KeyValuePair {
                    key: "dim".to_string(),
                    value: dimension.to_string(),
                }]
            }
            _ => vec![],
        };

        let data_type: milvus::proto::schema::DataType = field.ty.into();

        milvus::proto::schema::FieldSchema {
            field_id: unimplemented!(),
            name: field.name,
            is_primary_key,
            description: field.desc,
            data_type: data_type as i32,
            type_params,
            index_params: vec![],
            auto_id,
            state: milvus::proto::schema::FieldState::FieldCreated as _,
        }
    }
}
impl From<milvus::proto::schema::FieldSchema> for FieldSchema {
    fn from(field: milvus::proto::schema::FieldSchema) -> Self {
        let data_type = DataType::from_i32(field.data_type).unwrap();
        let ty = match data_type {
            milvus::proto::schema::DataType::None => FieldType::None,
            milvus::proto::schema::DataType::Bool => FieldType::Bool,
            milvus::proto::schema::DataType::Int8 => FieldType::Int8,
            milvus::proto::schema::DataType::Int16 => FieldType::Int16,
            milvus::proto::schema::DataType::Int32 => FieldType::Int32,
            milvus::proto::schema::DataType::Int64 => {
                FieldType::Int64(field.is_primary_key, field.auto_id)
            }
            milvus::proto::schema::DataType::Float => FieldType::Float,
            milvus::proto::schema::DataType::Double => FieldType::Double,
            milvus::proto::schema::DataType::String => FieldType::String,
            milvus::proto::schema::DataType::VarChar => FieldType::VarChar(
                field
                    .type_params
                    .iter()
                    .find(|kv| kv.key == "max_length")
                    .and_then(|kv| kv.value.parse().ok())
                    .unwrap(),
                field.is_primary_key,
                field.auto_id,
            ),
            milvus::proto::schema::DataType::BinaryVector => FieldType::BinaryVector(
                field
                    .type_params
                    .iter()
                    .find(|kv| kv.key == "dim")
                    .and_then(|kv| kv.value.parse().ok())
                    .unwrap(),
            ),
            milvus::proto::schema::DataType::FloatVector => FieldType::FloatVector(
                field
                    .type_params
                    .iter()
                    .find(|kv| kv.key == "dim")
                    .and_then(|kv| kv.value.parse().ok())
                    .unwrap(),
            ),
        };

        Self {
            name: field.name,
            desc: field.description,
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FieldType {
    None,
    Bool,
    Int8,
    Int16,
    Int32,
    /// `AutoId` is only valid when `PrimaryKey` is true.
    Int64(PrimaryKey, AutoId),
    Float,
    Double,
    String,
    /// `AutoId` is only valid when `PrimaryKey` is true.
    VarChar(MaxLength, PrimaryKey, AutoId),
    BinaryVector(Dimension),
    FloatVector(Dimension),
}
impl From<FieldType> for milvus::proto::schema::DataType {
    fn from(field_type: FieldType) -> Self {
        match field_type {
            FieldType::None => milvus::proto::schema::DataType::None,
            FieldType::Bool => milvus::proto::schema::DataType::Bool,
            FieldType::Int8 => milvus::proto::schema::DataType::Int8,
            FieldType::Int16 => milvus::proto::schema::DataType::Int16,
            FieldType::Int32 => milvus::proto::schema::DataType::Int32,
            FieldType::Int64(_, _) => milvus::proto::schema::DataType::Int64,
            FieldType::Float => milvus::proto::schema::DataType::Float,
            FieldType::Double => milvus::proto::schema::DataType::Double,
            FieldType::String => milvus::proto::schema::DataType::String,
            FieldType::VarChar(_, _, _) => milvus::proto::schema::DataType::VarChar,
            FieldType::BinaryVector(_) => milvus::proto::schema::DataType::BinaryVector,
            FieldType::FloatVector(_) => milvus::proto::schema::DataType::FloatVector,
        }
    }
}
impl From<milvus::proto::schema::DataType> for FieldType {
    fn from(data_type: milvus::proto::schema::DataType) -> Self {
        match data_type {
            milvus::proto::schema::DataType::None => FieldType::None,
            milvus::proto::schema::DataType::Bool => FieldType::Bool,
            milvus::proto::schema::DataType::Int8 => FieldType::Int8,
            milvus::proto::schema::DataType::Int16 => FieldType::Int16,
            milvus::proto::schema::DataType::Int32 => FieldType::Int32,
            milvus::proto::schema::DataType::Int64 => FieldType::Int64(false, false),
            milvus::proto::schema::DataType::Float => FieldType::Float,
            milvus::proto::schema::DataType::Double => FieldType::Double,
            milvus::proto::schema::DataType::String => FieldType::String,
            milvus::proto::schema::DataType::VarChar => FieldType::VarChar(0, false, false),
            milvus::proto::schema::DataType::BinaryVector => FieldType::BinaryVector(0),
            milvus::proto::schema::DataType::FloatVector => FieldType::FloatVector(0),
        }
    }
}
pub type AutoId = bool;
pub type PrimaryKey = bool;
pub type MaxLength = i32;
pub type Dimension = i64;

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
