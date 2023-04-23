use crate::common::{DataType, FieldState};
use num_traits::FromPrimitive;

#[derive(Debug, Clone, Default)]
pub struct CollectionSchema {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) auto_id: bool,
    pub(crate) fields: Vec<FieldSchema>,
}
impl From<CollectionSchema> for milvus::proto::schema::CollectionSchema {
    fn from(schema: CollectionSchema) -> Self {
        Self {
            name: schema.name.to_string(),
            description: schema.description,
            auto_id: schema.auto_id,
            fields: schema.fields.into_iter().map(Into::into).collect(),
        }
    }
}
impl From<milvus::proto::schema::CollectionSchema> for CollectionSchema {
    fn from(schema: milvus::proto::schema::CollectionSchema) -> Self {
        CollectionSchema {
            name: schema.name,
            description: schema.description,
            auto_id: schema.auto_id,
            fields: schema.fields.into_iter().map(Into::into).collect(),
        }
    }
}
impl CollectionSchema {
    pub fn new(name: &str, fields: Vec<FieldSchema>, description: Option<&str>) -> Self {
        Self {
            name: name.to_string(),
            description: description.unwrap_or_default().to_string(),
            fields,
            ..Default::default()
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fields(&self) -> &[FieldSchema] {
        &self.fields
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

#[derive(Debug, Clone, Default)]
pub struct FieldSchema {
    pub(crate) field_id: i64,
    pub(crate) name: String,
    pub(crate) is_primary_key: bool,
    pub(crate) description: String,
    pub(crate) data_type: DataType,
    pub(crate) type_params: std::collections::HashMap<String, String>,
    pub(crate) index_params: std::collections::HashMap<String, String>,
    pub(crate) auto_id: bool,
    /// To keep compatible with older version, the default state is `Created`.
    pub(crate) state: FieldState,
}
impl From<FieldSchema> for milvus::proto::schema::FieldSchema {
    fn from(field: FieldSchema) -> Self {
        Self {
            name: field.name,
            is_primary_key: field.is_primary_key,
            description: field.description,
            data_type: field.data_type as i32,
            type_params: field
                .type_params
                .into_iter()
                .map(|(k, v)| milvus::proto::common::KeyValuePair { key: k, value: v })
                .collect(),
            index_params: field
                .index_params
                .into_iter()
                .map(|(k, v)| milvus::proto::common::KeyValuePair { key: k, value: v })
                .collect(),
            auto_id: field.auto_id,
            state: field.state as i32,
            ..Default::default()
        }
    }
}
impl From<milvus::proto::schema::FieldSchema> for FieldSchema {
    fn from(field: milvus::proto::schema::FieldSchema) -> Self {
        Self {
            field_id: field.field_id,
            name: field.name,
            is_primary_key: field.is_primary_key,
            description: field.description,
            data_type: FromPrimitive::from_i32(field.data_type).unwrap(),
            type_params: field
                .type_params
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect(),
            index_params: field
                .index_params
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect(),
            auto_id: field.auto_id,
            state: FromPrimitive::from_i32(field.state).unwrap(),
        }
    }
}
impl FieldSchema {
    pub fn new(name: &str, ty: FieldType, description: Option<&str>) -> Self {
        let mut schema = FieldSchema::default();

        schema.name = name.to_string();
        schema.description = description.unwrap_or_default().to_string();
        schema.data_type = match ty {
            FieldType::None => DataType::None,
            FieldType::Bool => DataType::Bool,
            FieldType::Int8 => DataType::Int8,
            FieldType::Int16 => DataType::Int16,
            FieldType::Int32 => DataType::Int32,
            FieldType::Int64(pk, auto_id) => {
                schema.is_primary_key = pk;
                schema.auto_id = auto_id;
                DataType::Int64
            }
            FieldType::Float => DataType::Float,
            FieldType::Double => DataType::Double,
            FieldType::String => DataType::String,
            FieldType::VarChar(max_length, pk, auto_id) => {
                schema
                    .type_params
                    .insert("max_length".to_string(), max_length.to_string());
                schema.is_primary_key = pk;
                schema.auto_id = auto_id;
                DataType::VarChar
            }
            FieldType::BinaryVector(dim) => {
                schema
                    .type_params
                    .insert("dim".to_string(), dim.to_string());
                DataType::BinaryVector
            }
            FieldType::FloatVector(dim) => {
                schema
                    .type_params
                    .insert("dim".to_string(), dim.to_string());
                DataType::FloatVector
            }
        };

        schema
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

pub type AutoId = bool;
pub type PrimaryKey = bool;
pub type MaxLength = i32;
pub type Dimension = i64;
