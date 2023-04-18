use crate::error::VDBResult;
use std::fmt;

#[derive(Debug)]
pub struct Collection {
    pub(crate) inner: milvus::collection::Collection,
}

#[derive(Debug)]
pub struct CollectionSchema {
    inner: milvus::schema::CollectionSchema,
}
impl CollectionSchema {
    pub fn new(
        name: impl AsRef<str>,
        description: Option<&str>,
        fields: Vec<FieldSchema>,
    ) -> VDBResult<Self> {
        let mut builder = milvus::schema::CollectionSchemaBuilder::new(
            name.as_ref(),
            description.unwrap_or_default(),
        );

        if !fields.is_empty() {
            for field in fields {
                builder.add_field(field.into());
            }
        }

        match builder.build() {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(Box::new(e.into())),
        }
    }
}
impl From<CollectionSchema> for milvus::schema::CollectionSchema {
    fn from(schema: CollectionSchema) -> Self {
        schema.inner
    }
}

pub struct FieldSchema {
    // inner: milvus::schema::FieldSchema,
    name: String,
    desc: String,
    ty: FieldType,
}
impl FieldSchema {
    pub fn new(name: &str, description: &str, ty: FieldType) -> Self {
        Self {
            name: name.to_string(),
            desc: description.to_string(),
            ty,
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
            FieldType::Varchar(max_length, pk, auto_id) => {
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
impl From<milvus::schema::FieldSchema> for FieldSchema {
    fn from(field: milvus::schema::FieldSchema) -> Self {
        let ty = match field.dtype {
            milvus::proto::schema::DataType::None => FieldType::None,
            milvus::proto::schema::DataType::Bool => FieldType::Bool,
            milvus::proto::schema::DataType::Int8 => FieldType::Int8,
            milvus::proto::schema::DataType::Int16 => FieldType::Int16,
            milvus::proto::schema::DataType::Int32 => FieldType::Int32,
            milvus::proto::schema::DataType::Int64 => match field.is_primary {
                true => FieldType::Int64(true, field.auto_id),
                false => FieldType::Int64(false, field.auto_id),
            },
            milvus::proto::schema::DataType::Float => FieldType::Float,
            milvus::proto::schema::DataType::Double => FieldType::Double,
            milvus::proto::schema::DataType::String => FieldType::String,
            milvus::proto::schema::DataType::VarChar => match field.is_primary {
                true => FieldType::Varchar(field.max_length, true, field.auto_id),
                false => FieldType::Varchar(field.max_length, false, field.auto_id),
            },
            milvus::proto::schema::DataType::BinaryVector => FieldType::BinaryVector(field.dim),
            milvus::proto::schema::DataType::FloatVector => FieldType::FloatVector(field.dim),
        };

        Self {
            name: field.name,
            desc: field.description,
            ty,
        }
    }
}
impl From<FieldSchema> for milvus::schema::FieldSchema {
    fn from(field: FieldSchema) -> Self {
        match field.ty {
            FieldType::None => milvus::schema::FieldSchema::default(),
            FieldType::Bool => milvus::schema::FieldSchema::new_bool(&field.name, &field.desc),
            FieldType::Int8 => milvus::schema::FieldSchema::new_int8(&field.name, &field.desc),
            FieldType::Int16 => milvus::schema::FieldSchema::new_int16(&field.name, &field.desc),
            FieldType::Int32 => milvus::schema::FieldSchema::new_int32(&field.name, &field.desc),
            FieldType::Int64(pk, auto_id) => match pk {
                true => milvus::schema::FieldSchema::new_primary_int64(
                    &field.name,
                    &field.desc,
                    auto_id,
                ),
                false => milvus::schema::FieldSchema::new_int64(&field.name, &field.desc),
            },
            FieldType::Float => milvus::schema::FieldSchema::new_float(&field.name, &field.desc),
            FieldType::Double => milvus::schema::FieldSchema::new_double(&field.name, &field.desc),
            FieldType::String => milvus::schema::FieldSchema::new_string(&field.name, &field.desc),
            FieldType::Varchar(max_length, pk, auto_id) => match pk {
                true => milvus::schema::FieldSchema::new_primary_varchar(
                    &field.name,
                    &field.desc,
                    auto_id,
                    max_length,
                ),
                false => {
                    milvus::schema::FieldSchema::new_varchar(&field.name, &field.desc, max_length)
                }
            },
            FieldType::BinaryVector(dimension) => {
                milvus::schema::FieldSchema::new_binary_vector(&field.name, &field.desc, dimension)
            }
            FieldType::FloatVector(dimension) => {
                milvus::schema::FieldSchema::new_float_vector(&field.name, &field.desc, dimension)
            }
        }
    }
}

#[derive(Debug)]
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
    Varchar(MaxLength, PrimaryKey, AutoId),
    BinaryVector(Dimension),
    FloatVector(Dimension),
}

pub type AutoId = bool;
pub type PrimaryKey = bool;
pub type MaxLength = i32;
pub type Dimension = i64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_schema() {
        let field = FieldSchema::new("name", "desc", FieldType::Int64(true, true));
        println!("{:?}", field);
    }
}
