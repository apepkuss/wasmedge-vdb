use crate::{common::ConsistencyLevel, schema::CollectionSchema};

#[derive(Debug, Clone)]
pub struct CollectionMetadata {
    pub name: String,
    pub id: i64,
    /// The collection schema
    pub schema: Option<CollectionSchema>,
    /// Hybrid timestamp in milvus
    pub created_timestamp: u64,
    /// The utc timestamp calculated by created_timestamp
    pub created_utc_timestamp: u64,
    /// The shards number
    pub shards_num: i32,
    /// The aliases of this collection
    pub aliases: Vec<String>,
    /// The consistency level that the collection used
    pub consistency_level: ConsistencyLevel,
}

#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub name: String,
    pub id: i64,
    pub created_timestamp: u64,
    pub created_utc_timestamp: u64,
    pub in_memory_percentage: i64,
    pub query_service_available: bool,
}

#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub name: String,
    pub id: i64,
    pub created_timestamp: u64,
    pub created_utc_timestamp: u64,
    pub in_memory_percentage: i64,
}

#[derive(Debug, Clone)]
pub struct IndexInfo {
    pub index_name: String,
    pub index_id: i64,
    pub params: std::collections::HashMap<String, String>,
    pub field_name: String,
    pub indexed_rows: i64,
    pub total_rows: i64,
    pub state: i32,
    pub index_state_fail_reason: String,
}

#[derive(Debug, Clone)]
pub struct IndexState {
    pub state: i32,
    pub fail_reason: String,
}

#[derive(Debug, Clone)]
pub struct IndexProgress {
    pub indexed_rows: i64,
    pub total_rows: i64,
}

#[derive(Debug, Clone)]
pub struct FieldData {
    pub data_type: i32,
    pub field_name: String,
    pub field_id: i64,
    pub field: Option<Field>,
}
impl From<FieldData> for milvus::proto::schema::FieldData {
    fn from(field_data: FieldData) -> Self {
        Self {
            r#type: field_data.data_type,
            field_name: field_data.field_name,
            field_id: field_data.field_id,
            field: field_data.field.map(|f| f.into()),
        }
    }
}
impl From<milvus::proto::schema::FieldData> for FieldData {
    fn from(field_data: milvus::proto::schema::FieldData) -> Self {
        Self {
            data_type: field_data.r#type,
            field_name: field_data.field_name,
            field_id: field_data.field_id,
            field: field_data.field.map(|f| f.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Field {
    Scalars(ScalarField),
    Vectors(VectorField),
}
impl From<Field> for milvus::proto::schema::field_data::Field {
    fn from(field: Field) -> Self {
        match field {
            Field::Scalars(scalar_field) => milvus::proto::schema::field_data::Field::Scalars(
                milvus::proto::schema::ScalarField {
                    data: scalar_field.data.map(|data| data.into()),
                },
            ),
            Field::Vectors(vector_field) => milvus::proto::schema::field_data::Field::Vectors(
                milvus::proto::schema::VectorField {
                    dim: vector_field.dim,
                    data: vector_field.data.map(|data| data.into()),
                },
            ),
        }
    }
}
impl From<milvus::proto::schema::field_data::Field> for Field {
    fn from(field: milvus::proto::schema::field_data::Field) -> Self {
        match field {
            milvus::proto::schema::field_data::Field::Scalars(scalar_field) => {
                Field::Scalars(scalar_field.into())
            }
            milvus::proto::schema::field_data::Field::Vectors(vector_field) => {
                Field::Vectors(vector_field.into())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalarField {
    pub data: Option<ScalarFieldData>,
}
impl From<ScalarField> for milvus::proto::schema::ScalarField {
    fn from(field: ScalarField) -> Self {
        milvus::proto::schema::ScalarField {
            data: field.data.map(|data| data.into()),
        }
    }
}
impl From<milvus::proto::schema::ScalarField> for ScalarField {
    fn from(field: milvus::proto::schema::ScalarField) -> Self {
        Self {
            data: field.data.map(|data| data.into()),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarFieldData {
    BoolData(Vec<bool>),
    IntData(Vec<i32>),
    LongData(Vec<i64>),
    FloatData(Vec<f32>),
    DoubleData(Vec<f64>),
    StringData(Vec<String>),
    BytesData(Vec<Vec<u8>>),
}
impl From<ScalarFieldData> for milvus::proto::schema::scalar_field::Data {
    fn from(data: ScalarFieldData) -> Self {
        match data {
            ScalarFieldData::BoolData(v) => milvus::proto::schema::scalar_field::Data::BoolData({
                milvus::proto::schema::BoolArray { data: v }
            }),
            ScalarFieldData::IntData(v) => milvus::proto::schema::scalar_field::Data::IntData(
                milvus::proto::schema::IntArray { data: v },
            ),
            ScalarFieldData::LongData(v) => milvus::proto::schema::scalar_field::Data::LongData(
                milvus::proto::schema::LongArray { data: v },
            ),
            ScalarFieldData::FloatData(v) => milvus::proto::schema::scalar_field::Data::FloatData(
                milvus::proto::schema::FloatArray { data: v },
            ),
            ScalarFieldData::DoubleData(v) => {
                milvus::proto::schema::scalar_field::Data::DoubleData(
                    milvus::proto::schema::DoubleArray { data: v },
                )
            }
            ScalarFieldData::StringData(v) => {
                milvus::proto::schema::scalar_field::Data::StringData(
                    milvus::proto::schema::StringArray { data: v },
                )
            }
            ScalarFieldData::BytesData(v) => milvus::proto::schema::scalar_field::Data::BytesData(
                milvus::proto::schema::BytesArray { data: v },
            ),
        }
    }
}
impl From<milvus::proto::schema::scalar_field::Data> for ScalarFieldData {
    fn from(data: milvus::proto::schema::scalar_field::Data) -> Self {
        match data {
            milvus::proto::schema::scalar_field::Data::BoolData(v) => {
                ScalarFieldData::BoolData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::IntData(v) => {
                ScalarFieldData::IntData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::LongData(v) => {
                ScalarFieldData::LongData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::FloatData(v) => {
                ScalarFieldData::FloatData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::DoubleData(v) => {
                ScalarFieldData::DoubleData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::StringData(v) => {
                ScalarFieldData::StringData(v.data)
            }
            milvus::proto::schema::scalar_field::Data::BytesData(v) => {
                ScalarFieldData::BytesData(v.data)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorField {
    pub dim: i64,
    pub data: Option<VectorFieldData>,
}
impl From<VectorField> for milvus::proto::schema::VectorField {
    fn from(field: VectorField) -> Self {
        milvus::proto::schema::VectorField {
            dim: field.dim,
            data: field.data.map(|data| data.into()),
        }
    }
}
impl From<milvus::proto::schema::VectorField> for VectorField {
    fn from(field: milvus::proto::schema::VectorField) -> Self {
        VectorField {
            dim: field.dim,
            data: field.data.map(|data| data.into()),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum VectorFieldData {
    BinaryVec(Vec<u8>),
    FloatVec(Vec<f32>),
}
impl From<VectorFieldData> for milvus::proto::schema::vector_field::Data {
    fn from(data: VectorFieldData) -> Self {
        match data {
            VectorFieldData::BinaryVec(v) => {
                milvus::proto::schema::vector_field::Data::BinaryVector(v)
            }
            VectorFieldData::FloatVec(v) => milvus::proto::schema::vector_field::Data::FloatVector(
                milvus::proto::schema::FloatArray { data: v },
            ),
        }
    }
}
impl From<milvus::proto::schema::vector_field::Data> for VectorFieldData {
    fn from(data: milvus::proto::schema::vector_field::Data) -> Self {
        match data {
            milvus::proto::schema::vector_field::Data::BinaryVector(v) => {
                VectorFieldData::BinaryVec(v)
            }
            milvus::proto::schema::vector_field::Data::FloatVector(v) => {
                VectorFieldData::FloatVec(v.data)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MutationResult {
    pub id: Option<Id>,
    pub succ_index: Vec<u32>,
    pub err_index: Vec<u32>,
    pub acknowledged: bool,
    pub insert_cnt: i64,
    pub delete_cnt: i64,
    pub upsert_cnt: i64,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Id {
    id_field: Option<IdField>,
}
impl From<Id> for milvus::proto::schema::IDs {
    fn from(id: Id) -> Self {
        milvus::proto::schema::IDs {
            id_field: id.id_field.map(|id_field| id_field.into()),
        }
    }
}
impl From<milvus::proto::schema::IDs> for Id {
    fn from(ids: milvus::proto::schema::IDs) -> Self {
        Id {
            id_field: ids.id_field.map(|id_field| id_field.into()),
        }
    }
}
#[derive(Debug, Clone)]
pub enum IdField {
    IntId(Vec<i64>),
    StrId(Vec<String>),
}
impl From<IdField> for milvus::proto::schema::i_ds::IdField {
    fn from(id_field: IdField) -> Self {
        match id_field {
            IdField::IntId(v) => {
                milvus::proto::schema::i_ds::IdField::IntId(milvus::proto::schema::LongArray {
                    data: v,
                })
            }
            IdField::StrId(v) => {
                milvus::proto::schema::i_ds::IdField::StrId(milvus::proto::schema::StringArray {
                    data: v,
                })
            }
        }
    }
}
impl From<milvus::proto::schema::i_ds::IdField> for IdField {
    fn from(id_field: milvus::proto::schema::i_ds::IdField) -> Self {
        match id_field {
            milvus::proto::schema::i_ds::IdField::IntId(v) => IdField::IntId(v.data),
            milvus::proto::schema::i_ds::IdField::StrId(v) => IdField::StrId(v.data),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub results: Option<SearchResultData>,
    pub collection_name: String,
}

#[derive(Debug, Clone)]
pub struct SearchResultData {
    pub num_queries: i64,
    pub top_k: i64,
    pub fields_data: Vec<FieldData>,
    pub scores: Vec<f32>,
    pub id: Option<Id>,
    pub topks: Vec<i64>,
}
impl From<milvus::proto::schema::SearchResultData> for SearchResultData {
    fn from(data: milvus::proto::schema::SearchResultData) -> Self {
        SearchResultData {
            num_queries: data.num_queries,
            top_k: data.top_k,
            fields_data: data
                .fields_data
                .into_iter()
                .map(|field_data| field_data.into())
                .collect(),
            scores: data.scores,
            id: data.ids.map(|id| id.into()),
            topks: data.topks,
        }
    }
}
