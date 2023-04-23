use crate::{proto, schema::CollectionSchema};
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
impl From<ConsistencyLevel> for proto::common::ConsistencyLevel {
    fn from(level: ConsistencyLevel) -> Self {
        proto::common::ConsistencyLevel::from_i32(level.to_i32().unwrap()).unwrap()
    }
}
impl From<proto::common::ConsistencyLevel> for ConsistencyLevel {
    fn from(level: proto::common::ConsistencyLevel) -> Self {
        ConsistencyLevel::from_i32(level as i32).unwrap()
    }
}

// #[derive(Debug, Clone, Default)]
// pub struct CollectionSchema {
//     pub(crate) name: String,
//     pub(crate) description: String,
//     pub(crate) auto_id: bool,
//     pub(crate) fields: Vec<FieldSchema>,
// }
// impl From<CollectionSchema> for proto::schema::CollectionSchema {
//     fn from(schema: CollectionSchema) -> Self {
//         Self {
//             name: schema.name.to_string(),
//             description: schema.description,
//             auto_id: schema.auto_id,
//             fields: schema.fields.into_iter().map(Into::into).collect(),
//         }
//     }
// }
// impl From<proto::schema::CollectionSchema> for CollectionSchema {
//     fn from(schema: proto::schema::CollectionSchema) -> Self {
//         CollectionSchema {
//             name: schema.name,
//             description: schema.description,
//             auto_id: schema.auto_id,
//             fields: schema.fields.into_iter().map(Into::into).collect(),
//         }
//     }
// }
// impl CollectionSchema {
//     pub fn new(name: &str, fields: Vec<FieldSchema>, description: Option<&str>) -> Self {
//         Self {
//             name: name.to_string(),
//             description: description.unwrap_or_default().to_string(),
//             fields,
//             ..Default::default()
//         }
//     }

//     pub fn name(&self) -> &str {
//         &self.name
//     }

//     pub fn fields(&self) -> &[FieldSchema] {
//         &self.fields
//     }

//     pub fn description(&self) -> &str {
//         &self.description
//     }
// }

// #[derive(Debug, Clone, Default)]
// pub struct FieldSchema {
//     pub(crate) field_id: i64,
//     pub(crate) name: String,
//     pub(crate) is_primary_key: bool,
//     pub(crate) description: String,
//     pub(crate) data_type: DataType,
//     pub(crate) type_params: std::collections::HashMap<String, String>,
//     pub(crate) index_params: std::collections::HashMap<String, String>,
//     pub(crate) auto_id: bool,
//     /// To keep compatible with older version, the default state is `Created`.
//     pub(crate) state: FieldState,
// }
// impl From<FieldSchema> for proto::schema::FieldSchema {
//     fn from(field: FieldSchema) -> Self {
//         Self {
//             name: field.name,
//             is_primary_key: field.is_primary_key,
//             description: field.description,
//             data_type: field.data_type as i32,
//             type_params: field
//                 .type_params
//                 .into_iter()
//                 .map(|(k, v)| proto::common::KeyValuePair { key: k, value: v })
//                 .collect(),
//             index_params: field
//                 .index_params
//                 .into_iter()
//                 .map(|(k, v)| proto::common::KeyValuePair { key: k, value: v })
//                 .collect(),
//             auto_id: field.auto_id,
//             state: field.state as i32,
//             ..Default::default()
//         }
//     }
// }
// impl From<proto::schema::FieldSchema> for FieldSchema {
//     fn from(field: proto::schema::FieldSchema) -> Self {
//         Self {
//             field_id: field.field_id,
//             name: field.name,
//             is_primary_key: field.is_primary_key,
//             description: field.description,
//             data_type: FromPrimitive::from_i32(field.data_type).unwrap(),
//             type_params: field
//                 .type_params
//                 .into_iter()
//                 .map(|kv| (kv.key, kv.value))
//                 .collect(),
//             index_params: field
//                 .index_params
//                 .into_iter()
//                 .map(|kv| (kv.key, kv.value))
//                 .collect(),
//             auto_id: field.auto_id,
//             state: FromPrimitive::from_i32(field.state).unwrap(),
//         }
//     }
// }
// impl FieldSchema {
//     pub fn new(name: &str, ty: FieldType, description: Option<&str>) -> Self {
//         let mut schema = FieldSchema::default();

//         schema.name = name.to_string();
//         schema.description = description.unwrap_or_default().to_string();
//         schema.data_type = match ty {
//             FieldType::None => DataType::None,
//             FieldType::Bool => DataType::Bool,
//             FieldType::Int8 => DataType::Int8,
//             FieldType::Int16 => DataType::Int16,
//             FieldType::Int32 => DataType::Int32,
//             FieldType::Int64(pk, auto_id) => {
//                 schema.is_primary_key = pk;
//                 schema.auto_id = auto_id;
//                 DataType::Int64
//             }
//             FieldType::Float => DataType::Float,
//             FieldType::Double => DataType::Double,
//             FieldType::String => DataType::String,
//             FieldType::VarChar(max_length, pk, auto_id) => {
//                 schema
//                     .type_params
//                     .insert("max_length".to_string(), max_length.to_string());
//                 schema.is_primary_key = pk;
//                 schema.auto_id = auto_id;
//                 DataType::VarChar
//             }
//             FieldType::BinaryVector(dim) => {
//                 schema
//                     .type_params
//                     .insert("dim".to_string(), dim.to_string());
//                 DataType::BinaryVector
//             }
//             FieldType::FloatVector(dim) => {
//                 schema
//                     .type_params
//                     .insert("dim".to_string(), dim.to_string());
//                 DataType::FloatVector
//             }
//         };

//         schema
//     }
// }

// #[derive(Debug, Clone)]
// pub enum FieldType {
//     None,
//     Bool,
//     Int8,
//     Int16,
//     Int32,
//     /// `AutoId` is only valid when `PrimaryKey` is true.
//     Int64(PrimaryKey, AutoId),
//     Float,
//     Double,
//     String,
//     /// `AutoId` is only valid when `PrimaryKey` is true.
//     VarChar(MaxLength, PrimaryKey, AutoId),
//     BinaryVector(Dimension),
//     FloatVector(Dimension),
// }

// pub type AutoId = bool;
// pub type PrimaryKey = bool;
// pub type MaxLength = i32;
// pub type Dimension = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum DataType {
    None = 0,
    Bool = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    Float = 10,
    Double = 11,
    String = 20,
    /// variable-length strings with a specified maximum length
    VarChar = 21,
    BinaryVector = 100,
    FloatVector = 101,
}
impl Default for DataType {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum FieldState {
    FieldCreated = 0,
    FieldCreating = 1,
    FieldDropping = 2,
    FieldDropped = 3,
}
impl Default for FieldState {
    fn default() -> Self {
        Self::FieldCreated
    }
}

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
    // /// Load percentage on querynode when type is InMemory
    // /// Deprecated: use GetLoadingProgress rpc instead
    // pub in_memory_percentage: i64,
    // /// Indicate whether query service is available.
    // pub query_service_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum ShowType {
    /// Will return all collections
    All = 0,
    /// Will return loaded collections with their inMemory_percentages
    InMemory = 1,
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
impl From<FieldData> for proto::schema::FieldData {
    fn from(field_data: FieldData) -> Self {
        Self {
            r#type: field_data.data_type,
            field_name: field_data.field_name,
            field_id: field_data.field_id,
            field: field_data.field.map(|f| f.into()),
        }
    }
}
impl From<proto::schema::FieldData> for FieldData {
    fn from(field_data: proto::schema::FieldData) -> Self {
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
impl From<Field> for proto::schema::field_data::Field {
    fn from(field: Field) -> Self {
        match field {
            Field::Scalars(scalar_field) => {
                proto::schema::field_data::Field::Scalars(proto::schema::ScalarField {
                    data: scalar_field.data.map(|data| data.into()),
                })
            }
            Field::Vectors(vector_field) => {
                proto::schema::field_data::Field::Vectors(proto::schema::VectorField {
                    dim: vector_field.dim,
                    data: vector_field.data.map(|data| data.into()),
                })
            }
        }
    }
}
impl From<proto::schema::field_data::Field> for Field {
    fn from(field: proto::schema::field_data::Field) -> Self {
        match field {
            proto::schema::field_data::Field::Scalars(scalar_field) => {
                Field::Scalars(scalar_field.into())
            }
            proto::schema::field_data::Field::Vectors(vector_field) => {
                Field::Vectors(vector_field.into())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalarField {
    pub data: Option<ScalarFieldData>,
}
impl From<ScalarField> for proto::schema::ScalarField {
    fn from(field: ScalarField) -> Self {
        proto::schema::ScalarField {
            data: field.data.map(|data| data.into()),
        }
    }
}
impl From<proto::schema::ScalarField> for ScalarField {
    fn from(field: proto::schema::ScalarField) -> Self {
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
impl From<ScalarFieldData> for proto::schema::scalar_field::Data {
    fn from(data: ScalarFieldData) -> Self {
        match data {
            ScalarFieldData::BoolData(v) => proto::schema::scalar_field::Data::BoolData({
                proto::schema::BoolArray { data: v }
            }),
            ScalarFieldData::IntData(v) => {
                proto::schema::scalar_field::Data::IntData(proto::schema::IntArray { data: v })
            }
            ScalarFieldData::LongData(v) => {
                proto::schema::scalar_field::Data::LongData(proto::schema::LongArray { data: v })
            }
            ScalarFieldData::FloatData(v) => {
                proto::schema::scalar_field::Data::FloatData(proto::schema::FloatArray { data: v })
            }
            ScalarFieldData::DoubleData(v) => {
                proto::schema::scalar_field::Data::DoubleData(proto::schema::DoubleArray {
                    data: v,
                })
            }
            ScalarFieldData::StringData(v) => {
                proto::schema::scalar_field::Data::StringData(proto::schema::StringArray {
                    data: v,
                })
            }
            ScalarFieldData::BytesData(v) => {
                proto::schema::scalar_field::Data::BytesData(proto::schema::BytesArray { data: v })
            }
        }
    }
}
impl From<proto::schema::scalar_field::Data> for ScalarFieldData {
    fn from(data: proto::schema::scalar_field::Data) -> Self {
        match data {
            proto::schema::scalar_field::Data::BoolData(v) => ScalarFieldData::BoolData(v.data),
            proto::schema::scalar_field::Data::IntData(v) => ScalarFieldData::IntData(v.data),
            proto::schema::scalar_field::Data::LongData(v) => ScalarFieldData::LongData(v.data),
            proto::schema::scalar_field::Data::FloatData(v) => ScalarFieldData::FloatData(v.data),
            proto::schema::scalar_field::Data::DoubleData(v) => ScalarFieldData::DoubleData(v.data),
            proto::schema::scalar_field::Data::StringData(v) => ScalarFieldData::StringData(v.data),
            proto::schema::scalar_field::Data::BytesData(v) => ScalarFieldData::BytesData(v.data),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorField {
    pub dim: i64,
    pub data: Option<VectorFieldData>,
}
impl From<VectorField> for proto::schema::VectorField {
    fn from(field: VectorField) -> Self {
        proto::schema::VectorField {
            dim: field.dim,
            data: field.data.map(|data| data.into()),
        }
    }
}
impl From<proto::schema::VectorField> for VectorField {
    fn from(field: proto::schema::VectorField) -> Self {
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
impl From<VectorFieldData> for proto::schema::vector_field::Data {
    fn from(data: VectorFieldData) -> Self {
        match data {
            VectorFieldData::BinaryVec(v) => proto::schema::vector_field::Data::BinaryVector(v),
            VectorFieldData::FloatVec(v) => {
                proto::schema::vector_field::Data::FloatVector(proto::schema::FloatArray {
                    data: v,
                })
            }
        }
    }
}
impl From<proto::schema::vector_field::Data> for VectorFieldData {
    fn from(data: proto::schema::vector_field::Data) -> Self {
        match data {
            proto::schema::vector_field::Data::BinaryVector(v) => VectorFieldData::BinaryVec(v),
            proto::schema::vector_field::Data::FloatVector(v) => VectorFieldData::FloatVec(v.data),
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
impl From<Id> for proto::schema::IDs {
    fn from(id: Id) -> Self {
        proto::schema::IDs {
            id_field: id.id_field.map(|id_field| id_field.into()),
        }
    }
}
impl From<proto::schema::IDs> for Id {
    fn from(ids: proto::schema::IDs) -> Self {
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
impl From<IdField> for proto::schema::i_ds::IdField {
    fn from(id_field: IdField) -> Self {
        match id_field {
            IdField::IntId(v) => {
                proto::schema::i_ds::IdField::IntId(proto::schema::LongArray { data: v })
            }
            IdField::StrId(v) => {
                proto::schema::i_ds::IdField::StrId(proto::schema::StringArray { data: v })
            }
        }
    }
}
impl From<proto::schema::i_ds::IdField> for IdField {
    fn from(id_field: proto::schema::i_ds::IdField) -> Self {
        match id_field {
            proto::schema::i_ds::IdField::IntId(v) => IdField::IntId(v.data),
            proto::schema::i_ds::IdField::StrId(v) => IdField::StrId(v.data),
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
impl From<proto::schema::SearchResultData> for SearchResultData {
    fn from(data: proto::schema::SearchResultData) -> Self {
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

#[derive(Debug, Clone)]
pub struct FlushResult {
    pub db_name: String,
    pub collection_segment_ids: std::collections::HashMap<String, Vec<i64>>,
    pub flush_collection_segment_ids: std::collections::HashMap<String, Vec<i64>>,
    pub collection_seal_times: std::collections::HashMap<String, i64>,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub fields_data: Vec<FieldData>,
    pub collection_name: String,
}

#[derive(Debug, Clone)]
pub struct PersistentSegmentInfo {
    pub segment_id: i64,
    pub collection_id: i64,
    pub partition_id: i64,
    pub num_rows: i64,
    pub state: SegmentState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
pub enum SegmentState {
    None = 0,
    NotExist = 1,
    Growing = 2,
    Sealed = 3,
    Flushed = 4,
    Flushing = 5,
    Dropped = 6,
    Importing = 7,
}

#[derive(Debug, Clone)]
pub struct QuerySegmentInfo {
    pub segment_id: i64,
    pub collection_id: i64,
    pub partition_id: i64,
    pub mem_size: i64,
    pub num_rows: i64,
    pub index_name: String,
    pub index_id: i64,
    /// deprecated, check node_ids(NodeIds) field
    pub node_id: i64,
    pub state: SegmentState,
    pub node_ids: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    pub replica_id: i64,
    pub collection_id: i64,
    /// empty indicates to load collection
    pub partition_ids: Vec<i64>,
    pub shard_replicas: Vec<ShardReplica>,
    /// include leaders
    pub node_ids: Vec<i64>,
}
impl From<proto::milvus::ReplicaInfo> for ReplicaInfo {
    fn from(replica_info: proto::milvus::ReplicaInfo) -> Self {
        ReplicaInfo {
            replica_id: replica_info.replica_id,
            collection_id: replica_info.collection_id,
            partition_ids: replica_info.partition_ids,
            shard_replicas: replica_info
                .shard_replicas
                .into_iter()
                .map(|shard_replica| shard_replica.into())
                .collect(),
            node_ids: replica_info.node_ids,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShardReplica {
    pub leader_id: i64,
    /// IP:port
    pub leader_addr: String,
    pub dm_channel_name: String,
    /// optional, DO NOT save it in meta, set it only for GetReplicas()
    /// if with_shard_nodes is true
    pub node_ids: Vec<i64>,
}
impl From<proto::milvus::ShardReplica> for ShardReplica {
    fn from(shard_replica: proto::milvus::ShardReplica) -> Self {
        ShardReplica {
            leader_id: shard_replica.leader_id,
            leader_addr: shard_replica.leader_addr,
            dm_channel_name: shard_replica.dm_channel_name,
            node_ids: shard_replica.node_ids,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Address {
    pub ip: String,
    pub port: i64,
}
impl From<proto::common::Address> for Address {
    fn from(address: proto::common::Address) -> Self {
        Address {
            ip: address.ip,
            port: address.port,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Metrics {
    /// response is of jsonic format
    pub response: String,
    /// metrics from which component
    pub component_name: String,
}

#[derive(Debug, Clone)]
pub struct ComponentState {
    pub state: Option<ComponentInfo>,
    pub subcomponent_states: Vec<ComponentInfo>,
}

#[derive(Debug, Clone)]
pub struct ComponentInfo {
    pub node_id: i64,
    pub role: ::prost::alloc::string::String,
    pub state_code: StateCode,
    pub extra_info: std::collections::HashMap<String, String>,
}
impl From<proto::milvus::ComponentInfo> for ComponentInfo {
    fn from(component_info: proto::milvus::ComponentInfo) -> Self {
        ComponentInfo {
            node_id: component_info.node_id,
            role: component_info.role,
            state_code: StateCode::from_i32(component_info.state_code).unwrap(),
            extra_info: component_info
                .extra_info
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum StateCode {
    Initializing = 0,
    Healthy = 1,
    Abnormal = 2,
    StandBy = 3,
}

#[derive(Debug, Clone)]
pub struct CompactionStateResult {
    pub state: CompactionState,
    pub executing_plan_no: i64,
    pub timeout_plan_no: i64,
    pub completed_plan_no: i64,
    pub failed_plan_no: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
pub enum CompactionState {
    UndefiedState = 0,
    Executing = 1,
    Completed = 2,
}

#[derive(Debug, Clone)]
pub struct CompactionPlan {
    pub state: CompactionState,
    pub merge_infos: Vec<CompactionMergeInfo>,
}

#[derive(Debug, Clone)]
pub struct CompactionMergeInfo {
    pub sources: Vec<i64>,
    pub target: i64,
}

#[derive(Debug, Clone)]
pub struct ImportStateResult {
    pub state: ImportState,
    pub row_count: i64,
    pub id_list: Vec<i64>,
    pub infos: std::collections::HashMap<String, String>,
    pub id: i64,
    pub collection_id: i64,
    pub segment_ids: Vec<i64>,
    pub create_ts: i64,
}
impl From<proto::milvus::GetImportStateResponse> for ImportStateResult {
    fn from(response: proto::milvus::GetImportStateResponse) -> Self {
        ImportStateResult {
            state: ImportState::from_i32(response.state).unwrap(),
            row_count: response.row_count,
            id_list: response.id_list,
            infos: response
                .infos
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect(),
            id: response.id,
            collection_id: response.collection_id,
            segment_ids: response.segment_ids,
            create_ts: response.create_ts,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
pub enum ImportState {
    /// the task in in pending list of rootCoord, waiting to be executed
    ImportPending = 0,
    /// the task failed for some reason, get detail reason from GetImportStateResponse.infos
    ImportFailed = 1,
    /// the task has been sent to datanode to execute
    ImportStarted = 2,
    /// all data files have been parsed and data already persisted
    ImportPersisted = 5,
    /// all indexes are successfully built and segments are able to be compacted as normal.
    ImportCompleted = 6,
    /// the task failed and all segments it generated are cleaned up.
    ImportFailedAndCleaned = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum OperateUserRoleType {
    AddUserToRole = 0,
    RemoveUserFromRole = 1,
}

#[derive(Debug, Clone)]
pub struct RoleResult {
    pub role: Option<RoleEntity>,
    pub users: Vec<UserEntity>,
}

#[derive(Debug, Clone)]
pub struct User {
    pub user: Option<UserEntity>,
    pub roles: Vec<RoleEntity>,
}
impl From<proto::milvus::UserResult> for User {
    fn from(user: proto::milvus::UserResult) -> Self {
        User {
            user: user.user.map(|user| user.into()),
            roles: user.roles.into_iter().map(|role| role.into()).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum OperatePrivilegeType {
    Grant = 0,
    Revoke = 1,
}

#[derive(Debug, Clone, Default)]
pub struct GrantEntity {
    pub role: Option<RoleEntity>,
    pub object: Option<ObjectEntity>,
    pub object_name: String,
    pub grantor: Option<GrantorEntity>,
}
impl From<proto::milvus::GrantEntity> for GrantEntity {
    fn from(grant_entity: proto::milvus::GrantEntity) -> Self {
        GrantEntity {
            role: grant_entity.role.map(|role| role.into()),
            object: grant_entity.object.map(|object| object.into()),
            object_name: grant_entity.object_name,
            grantor: grant_entity.grantor.map(|grantor| grantor.into()),
        }
    }
}
impl From<GrantEntity> for proto::milvus::GrantEntity {
    fn from(grant_entity: GrantEntity) -> Self {
        proto::milvus::GrantEntity {
            role: grant_entity.role.map(|role| role.into()),
            object: grant_entity.object.map(|object| object.into()),
            object_name: grant_entity.object_name,
            grantor: grant_entity.grantor.map(|grantor| grantor.into()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GrantorEntity {
    pub user: Option<UserEntity>,
    pub privilege: Option<PrivilegeEntity>,
}
impl From<proto::milvus::GrantorEntity> for GrantorEntity {
    fn from(grantor_entity: proto::milvus::GrantorEntity) -> Self {
        GrantorEntity {
            user: grantor_entity.user.map(|user| user.into()),
            privilege: grantor_entity.privilege.map(|privilege| privilege.into()),
        }
    }
}
impl From<GrantorEntity> for proto::milvus::GrantorEntity {
    fn from(grantor_entity: GrantorEntity) -> Self {
        proto::milvus::GrantorEntity {
            user: grantor_entity.user.map(|user| user.into()),
            privilege: grantor_entity.privilege.map(|privilege| privilege.into()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UserEntity {
    pub name: String,
}
impl From<proto::milvus::UserEntity> for UserEntity {
    fn from(user_entity: proto::milvus::UserEntity) -> Self {
        UserEntity {
            name: user_entity.name,
        }
    }
}
impl From<UserEntity> for proto::milvus::UserEntity {
    fn from(user_entity: UserEntity) -> Self {
        proto::milvus::UserEntity {
            name: user_entity.name,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PrivilegeEntity {
    pub name: String,
}
impl From<proto::milvus::PrivilegeEntity> for PrivilegeEntity {
    fn from(privilege_entity: proto::milvus::PrivilegeEntity) -> Self {
        PrivilegeEntity {
            name: privilege_entity.name,
        }
    }
}
impl From<PrivilegeEntity> for proto::milvus::PrivilegeEntity {
    fn from(privilege_entity: PrivilegeEntity) -> Self {
        proto::milvus::PrivilegeEntity {
            name: privilege_entity.name,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObjectEntity {
    pub name: String,
}
impl From<proto::milvus::ObjectEntity> for ObjectEntity {
    fn from(object_entity: proto::milvus::ObjectEntity) -> Self {
        ObjectEntity {
            name: object_entity.name,
        }
    }
}
impl From<ObjectEntity> for proto::milvus::ObjectEntity {
    fn from(object_entity: ObjectEntity) -> Self {
        proto::milvus::ObjectEntity {
            name: object_entity.name,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RoleEntity {
    pub name: String,
}
impl From<proto::milvus::RoleEntity> for RoleEntity {
    fn from(role_entity: proto::milvus::RoleEntity) -> Self {
        RoleEntity {
            name: role_entity.name,
        }
    }
}
impl From<RoleEntity> for proto::milvus::RoleEntity {
    fn from(role_entity: RoleEntity) -> Self {
        proto::milvus::RoleEntity {
            name: role_entity.name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Health {
    pub is_healthy: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, ToPrimitive)]
#[repr(i32)]
pub enum DslType {
    Dsl = 0,
    BoolExprV1 = 1,
}
