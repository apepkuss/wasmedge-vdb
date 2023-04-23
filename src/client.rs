use base64::engine::general_purpose;
use base64::Engine;
use num_traits::FromPrimitive;
use prost::{bytes::BytesMut, Message};
use tonic::codegen::InterceptedService;
use tonic::service::Interceptor;
use tonic::transport::Channel;
use tonic::Request;

use crate::{
    common::{
        Address, CollectionInfo, CollectionMetadata, CompactionMergeInfo, CompactionPlan,
        CompactionState, CompactionStateResult, ComponentState, ConsistencyLevel, DslType,
        FieldData, FlushResult, GrantEntity, Health, ImportState, ImportStateResult, IndexInfo,
        IndexProgress, IndexState, Metrics, MutationResult, OperatePrivilegeType,
        OperateUserRoleType, PartitionInfo, PersistentSegmentInfo, QueryResult, QuerySegmentInfo,
        ReplicaInfo, RoleEntity, RoleResult, SearchResult, SegmentState, ShowType, User,
        UserEntity,
    },
    error::{Error, Result},
    proto::{self, common::MsgType},
    schema::CollectionSchema,
    utils::{new_msg, status_to_result},
};

use std::collections::HashMap;

#[derive(Debug)]
pub struct Client {
    client: proto::milvus::milvus_service_client::MilvusServiceClient<
        InterceptedService<Channel, AuthInterceptor>,
    >,
}
impl Client {
    pub async fn new(
        host: &str,
        port: u16,
        username: Option<String>,
        password: Option<String>,
        timeout: Option<std::time::Duration>,
    ) -> Result<Self> {
        let url = format!("{}:{}", host, port.to_string());
        let timeout = match timeout {
            Some(timeout) => timeout,
            None => std::time::Duration::from_secs(10),
        };

        let mut dst: tonic::transport::Endpoint = url.try_into().map_err(|err| {
            Error::InvalidParameter("url".to_owned(), format!("to parse {:?}", err))
        })?;

        dst = dst.timeout(timeout);

        let token = match (username, password) {
            (Some(username), Some(password)) => {
                let auth_token = format!("{}:{}", username, password);
                let auth_token = general_purpose::STANDARD.encode(auth_token);
                Some(auth_token)
            }
            _ => None,
        };

        let auth_interceptor = AuthInterceptor { token };

        let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;

        let client = proto::milvus::milvus_service_client::MilvusServiceClient::with_interceptor(
            conn,
            auth_interceptor,
        );

        Ok(Self { client })
    }

    /// Create a collection with the specified schema.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The unique name of the collection to create.
    ///
    /// * `schema` - The schema of the collection to create.
    ///
    /// * `shards_num` - The shard number of the collection to create. It corresponds to the number of data nodes used to insert data.
    ///
    /// * `level` - The consistency level of the collection to create.
    ///
    /// * `properties` - The properties for modifying the collection.
    pub async fn create_collection(
        &self,
        collection_name: &str,
        schema: CollectionSchema,
        shards_num: Option<i32>,
        level: Option<ConsistencyLevel>,
        properties: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let schema: proto::schema::CollectionSchema = schema.into();
        let mut buf = BytesMut::new();
        schema.encode(&mut buf)?;
        let schema: Vec<u8> = buf.to_vec();

        let shards_num = shards_num.unwrap_or(2);

        let consistency_level = level.unwrap_or(ConsistencyLevel::Session);

        let properties = properties.unwrap_or_default();

        let request = proto::milvus::CreateCollectionRequest {
            base: Some(new_msg(MsgType::CreateCollection)),
            collection_name: collection_name.to_string(),
            schema,
            shards_num,
            consistency_level: consistency_level as i32,
            properties: properties
                .iter()
                .map(|(k, v)| proto::common::KeyValuePair {
                    key: k.to_string(),
                    value: v.to_string(),
                })
                .collect(),
            ..Default::default()
        };

        let status = self
            .client
            .clone()
            .create_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn drop_collection(&self, collection_name: &str) -> Result<()> {
        let request = proto::milvus::DropCollectionRequest {
            base: Some(new_msg(MsgType::DropCollection)),
            collection_name: collection_name.to_string(),
            ..Default::default()
        };

        let status = self
            .client
            .clone()
            .drop_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Check collection exist in milvus or not.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to check
    ///
    /// * `time_stamp` - The timestamp of the collection to check. If `time_stamp` is not zero, will return true when time_stamp >= created collection timestamp, otherwise will return false.
    ///
    pub async fn has_collection(
        &self,
        collection_name: &str,
        time_stamp: Option<u64>,
    ) -> Result<bool> {
        let request = proto::milvus::HasCollectionRequest {
            base: Some(new_msg(MsgType::HasCollection)),
            collection_name: collection_name.to_string(),
            time_stamp: time_stamp.unwrap_or(0),
            ..Default::default()
        };

        let response = self
            .client
            .clone()
            .has_collection(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.value)
    }

    /// Load collection data into query nodes, then you can do vector search on this collection.
    ///
    /// # Arguments
    ///
    /// * `db_name` - database name. Not useful for now.
    ///
    /// * `collection_name` - The name of the collection to load
    ///
    /// * `replica_num` - The number of replica to load. Default is 1.
    ///
    pub async fn load_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        replica_num: Option<i32>,
    ) -> Result<()> {
        let replica_number = replica_num.unwrap_or(1);

        let request = proto::milvus::LoadCollectionRequest {
            base: Some(new_msg(MsgType::LoadCollection)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            replica_number,
        };

        let status = self
            .client
            .clone()
            .load_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn release_collection(&self, db_name: &str, collection_name: &str) -> Result<()> {
        let request = proto::milvus::ReleaseCollectionRequest {
            base: Some(new_msg(MsgType::ReleaseCollection)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
        };

        let status = self
            .client
            .clone()
            .release_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Get collection meta datas like: schema, collectionID, shards number ...
    ///
    /// # Arguments
    ///
    /// * `name` - collection name
    ///
    pub async fn describe_collection(
        &self,
        collection_name: &str,
        time_stamp: Option<u64>,
    ) -> Result<CollectionMetadata> {
        let request = proto::milvus::DescribeCollectionRequest {
            base: Some(new_msg(MsgType::DescribeCollection)),
            collection_name: collection_name.to_string(),
            time_stamp: time_stamp.unwrap_or(0),
            ..Default::default()
        };

        let response = self
            .client
            .clone()
            .describe_collection(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let metadata = CollectionMetadata {
            name: response.collection_name,
            id: response.collection_id,
            schema: response.schema.map(|x| x.into()),
            created_timestamp: response.created_timestamp,
            created_utc_timestamp: response.created_utc_timestamp,
            shards_num: response.shards_num,
            aliases: response.aliases,
            consistency_level: crate::common::ConsistencyLevel::from_i32(
                response.consistency_level,
            )
            .unwrap(),
        };

        Ok(metadata)
    }

    /// Get collection statistics
    ///
    /// # Arguments
    ///
    /// * `name` - collection name
    ///
    pub async fn get_collection_stats(
        &self,
        db_name: &str,
        collection_name: &str,
    ) -> Result<HashMap<String, String>> {
        let request = proto::milvus::GetCollectionStatisticsRequest {
            base: Some(new_msg(MsgType::GetCollectionStatistics)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_collection_statistics(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let stats: HashMap<String, String> =
            HashMap::from_iter(response.stats.into_iter().map(|x| (x.key, x.value)));

        Ok(stats)
    }

    /// Return basic collection infos.
    pub async fn show_collections(
        &self,
        collection_names: Vec<&str>,
    ) -> Result<Vec<CollectionInfo>> {
        let request = proto::milvus::ShowCollectionsRequest {
            base: Some(new_msg(MsgType::ShowCollections)),
            collection_names: collection_names.iter().map(|x| x.to_string()).collect(),
            ..Default::default()
        };

        let response = self
            .client
            .clone()
            .show_collections(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let mut info_vec = vec![];
        for i in 0..response.collection_names.len() {
            info_vec.push(CollectionInfo {
                name: response.collection_names[i].clone(),
                id: response.collection_ids[i],
                created_timestamp: response.created_timestamps[i],
                created_utc_timestamp: response.created_utc_timestamps[i],
                // TODO: add in_memory_percentage and query_service_available
                // in_memory_percentage: response.in_memory_percentages[i],
                // query_service_available: response.query_service_available[i],
            });
        }

        Ok(info_vec)
    }

    /// Alter collection.
    pub async fn alter_collection(
        &self,
        db_name: &str,
        collection_name: &str,
        collection_id: i64,
        properties: Vec<(String, String)>,
    ) -> Result<()> {
        let request = proto::milvus::AlterCollectionRequest {
            base: Some(new_msg(MsgType::AlterCollection)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            collection_id,
            properties: properties
                .into_iter()
                .map(|(key, value)| proto::common::KeyValuePair { key, value })
                .collect(),
        };

        let status = self
            .client
            .clone()
            .alter_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Create partition in created collection.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the existed collection in which to create the partition.
    ///
    /// * `partition_name` - The name of the partition to create
    pub async fn create_partition(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
    ) -> Result<()> {
        let request = proto::milvus::CreatePartitionRequest {
            base: Some(new_msg(MsgType::CreatePartition)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
        };

        let status = self
            .client
            .clone()
            .create_partition(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Drop partition in created collection.
    pub async fn drop_partition(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
    ) -> Result<()> {
        let request = proto::milvus::DropPartitionRequest {
            base: Some(new_msg(MsgType::DropPartition)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
        };

        let status = self
            .client
            .clone()
            .drop_partition(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Check if partition exist in collection or not.
    pub async fn has_partition(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
    ) -> Result<bool> {
        let request = proto::milvus::HasPartitionRequest {
            base: Some(new_msg(MsgType::HasPartition)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .has_partition(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.value)
    }

    /// Load specific partitions data of one collection into query nodes
    /// Then you can get these data as result when you do vector search on this collection.
    pub async fn load_partitions(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_names: Vec<&str>,
        replica_number: i32,
    ) -> Result<()> {
        let request = proto::milvus::LoadPartitionsRequest {
            base: Some(new_msg(MsgType::LoadPartitions)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.iter().map(|x| x.to_string()).collect(),
            replica_number,
        };

        let status = self
            .client
            .clone()
            .load_partitions(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Release specific partitions data of one collection from query nodes.
    /// Then you can not get these data as result when you do vector search on this collection.
    pub async fn release_partitions(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_names: Vec<&str>,
    ) -> Result<()> {
        let request = proto::milvus::ReleasePartitionsRequest {
            base: Some(new_msg(MsgType::ReleasePartitions)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.iter().map(|x| x.to_string()).collect(),
        };

        let status = self
            .client
            .clone()
            .release_partitions(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Get partition statistics.
    pub async fn get_partition_stats(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
    ) -> Result<HashMap<String, String>> {
        let request = proto::milvus::GetPartitionStatisticsRequest {
            base: Some(new_msg(MsgType::GetPartitionStatistics)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_partition_statistics(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let stats: HashMap<String, String> =
            HashMap::from_iter(response.stats.into_iter().map(|x| (x.key, x.value)));

        Ok(stats)
    }

    /// List all partitions for particular collection.
    pub async fn show_partitions(
        &self,
        db_name: &str,
        collection_name: &str,
        collection_id: i64,
        partition_names: Option<Vec<&str>>,
        ty: ShowType,
    ) -> Result<Vec<PartitionInfo>> {
        let request = proto::milvus::ShowPartitionsRequest {
            base: Some(new_msg(MsgType::ShowPartitions)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            collection_id,
            partition_names: partition_names
                .unwrap_or_default()
                .iter()
                .map(|x| x.to_string())
                .collect(),
            r#type: ty as i32,
        };

        let response = self
            .client
            .clone()
            .show_partitions(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let mut res = vec![];
        for i in 0..response.partition_names.len() {
            res.push(PartitionInfo {
                name: response.partition_names[i].clone(),
                id: response.partition_i_ds[i],
                created_timestamp: response.created_timestamps[i],
                created_utc_timestamp: response.created_utc_timestamps[i],
                in_memory_percentage: response.in_memory_percentages[i],
            });
        }

        Ok(res)
    }

    pub async fn get_loading_progress(
        &self,
        collection_name: &str,
        partition_names: Vec<&str>,
    ) -> Result<i64> {
        let request = proto::milvus::GetLoadingProgressRequest {
            base: Some(new_msg(MsgType::LoadPartitions)),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.iter().map(|x| x.to_string()).collect(),
        };

        let response = self
            .client
            .clone()
            .get_loading_progress(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.progress)
    }

    pub async fn create_alias(
        &self,
        db_name: &str,
        collection_name: &str,
        alias: &str,
    ) -> Result<()> {
        let request = proto::milvus::CreateAliasRequest {
            base: Some(new_msg(MsgType::CreateAlias)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            alias: alias.to_string(),
        };

        let status = self
            .client
            .clone()
            .create_alias(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn drop_alias(&self, db_name: &str, alias: &str) -> Result<()> {
        let request = proto::milvus::DropAliasRequest {
            base: Some(new_msg(MsgType::DropAlias)),
            db_name: db_name.to_string(),
            alias: alias.to_string(),
        };

        let status = self.client.clone().drop_alias(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn alter_alias(
        &self,
        db_name: &str,
        collection_name: &str,
        alias: &str,
    ) -> Result<()> {
        let request = proto::milvus::AlterAliasRequest {
            base: Some(new_msg(MsgType::AlterAlias)),
            db_name: db_name.to_string(),
            alias: alias.to_string(),
            collection_name: collection_name.to_string(),
        };

        let status = self.client.clone().alter_alias(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    /// Create index for vector data
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to create index.
    ///
    /// * `field_name` - The name of the vector field in the collection.
    ///
    ///
    ///
    /// * `index` - The index to create.
    pub async fn create_index(
        &self,
        db_name: &str,
        collection_name: &str,
        field_name: &str,
        extra_params: Option<HashMap<String, String>>,
        index_name: Option<&str>,
    ) -> Result<()> {
        let request = proto::milvus::CreateIndexRequest {
            base: Some(new_msg(MsgType::CreateIndex)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            extra_params: extra_params
                .unwrap_or_default()
                .iter()
                .map(|(key, value)| proto::common::KeyValuePair {
                    key: key.clone(),
                    value: value.clone(),
                })
                .collect(),
            index_name: index_name.unwrap_or_default().to_string(),
        };

        let status = self
            .client
            .clone()
            .create_index(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn describe_index(
        &self,
        db_name: &str,
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<Vec<IndexInfo>> {
        let request = proto::milvus::DescribeIndexRequest {
            base: Some(new_msg(MsgType::DescribeIndex)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .describe_index(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let mut res = vec![];
        for i in 0..response.index_descriptions.len() {
            res.push(IndexInfo {
                index_name: response.index_descriptions[i].index_name.clone(),
                index_id: response.index_descriptions[i].index_id,
                params: response.index_descriptions[i]
                    .params
                    .iter()
                    .map(|kv| (kv.key.clone(), kv.value.clone()))
                    .collect(),
                field_name: response.index_descriptions[i].field_name.clone(),
                indexed_rows: response.index_descriptions[i].indexed_rows,
                total_rows: response.index_descriptions[i].total_rows,
                state: response.index_descriptions[i].state,
                index_state_fail_reason: response.index_descriptions[i]
                    .index_state_fail_reason
                    .clone(),
            });
        }

        Ok(res)
    }

    pub async fn get_index_state(
        &self,
        db_name: &str,
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<IndexState> {
        let request = proto::milvus::GetIndexStateRequest {
            base: Some(new_msg(MsgType::GetIndexState)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_index_state(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(IndexState {
            state: response.state,
            fail_reason: response.fail_reason,
        })
    }

    pub async fn get_index_build_progress(
        &self,
        db_name: &str,
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<IndexProgress> {
        let request = proto::milvus::GetIndexBuildProgressRequest {
            base: Some(new_msg(MsgType::GetIndexBuildProgress)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_index_build_progress(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(IndexProgress {
            total_rows: response.total_rows,
            indexed_rows: response.indexed_rows,
        })
    }

    pub async fn drop_index(
        &self,
        db_name: &str,
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<()> {
        let request = proto::milvus::DropIndexRequest {
            base: Some(new_msg(MsgType::DropIndex)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
        };

        let status = self.client.clone().drop_index(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn insert(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
        fields_data: Vec<FieldData>,
        hash_keys: Vec<u32>,
        num_rows: u32,
    ) -> Result<MutationResult> {
        let request = proto::milvus::InsertRequest {
            base: Some(new_msg(MsgType::Insert)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            fields_data: fields_data
                .into_iter()
                .map(|field_data| field_data.into())
                .collect(),
            hash_keys,
            num_rows,
        };

        let response = self.client.clone().insert(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = MutationResult {
            id: response.i_ds.map(|ids| ids.into()),
            succ_index: response.succ_index,
            err_index: response.err_index,
            acknowledged: response.acknowledged,
            insert_cnt: response.insert_cnt,
            delete_cnt: response.delete_cnt,
            upsert_cnt: response.upsert_cnt,
            timestamp: response.timestamp,
        };

        Ok(res)
    }

    pub async fn delete(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_name: &str,
        expr: &str,
        hash_keys: Vec<u32>,
    ) -> Result<MutationResult> {
        let request = proto::milvus::DeleteRequest {
            base: Some(new_msg(MsgType::Delete)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            expr: expr.to_string(),
            hash_keys,
        };

        let response = self.client.clone().delete(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = MutationResult {
            id: response.i_ds.map(|ids| ids.into()),
            succ_index: response.succ_index,
            err_index: response.err_index,
            acknowledged: response.acknowledged,
            insert_cnt: response.insert_cnt,
            delete_cnt: response.delete_cnt,
            upsert_cnt: response.upsert_cnt,
            timestamp: response.timestamp,
        };

        Ok(res)
    }

    pub async fn search(
        &self,
        db_name: &str,
        collection_name: &str,
        partition_names: Vec<&str>,
        dsl: &str,
        placeholder_group: Vec<u8>,
        dsl_type: DslType,
        output_fields: Vec<String>,
        search_params: HashMap<String, String>,
        travel_timestamp: u64,
        guarantee_timestamp: u64,
        nq: i64,
    ) -> Result<SearchResult> {
        let request = proto::milvus::SearchRequest {
            base: Some(new_msg(MsgType::Search)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.into_iter().map(|s| s.to_string()).collect(),
            dsl: dsl.to_string(),
            placeholder_group,
            dsl_type: dsl_type as i32,
            output_fields,
            search_params: search_params
                .into_iter()
                .map(|(k, v)| proto::common::KeyValuePair {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            travel_timestamp,
            guarantee_timestamp,
            nq,
        };

        let response = self.client.clone().search(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = SearchResult {
            results: response.results.map(|x| x.into()),
            collection_name: response.collection_name,
        };

        Ok(res)
    }

    pub async fn flush(&self, db_name: &str, collection_names: Vec<&str>) -> Result<FlushResult> {
        let request = proto::milvus::FlushRequest {
            base: Some(new_msg(MsgType::Flush)),
            db_name: db_name.to_string(),
            collection_names: collection_names
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        };

        let response = self.client.clone().flush(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = FlushResult {
            db_name: response.db_name,
            collection_segment_ids: response
                .coll_seg_i_ds
                .into_iter()
                .map(|(key, value)| (key, value.data))
                .collect(),
            flush_collection_segment_ids: response
                .flush_coll_seg_i_ds
                .into_iter()
                .map(|(key, value)| (key, value.data))
                .collect(),
            collection_seal_times: response.coll_seal_times,
        };

        Ok(res)
    }

    pub async fn query(
        &self,
        db_name: &str,
        collection_name: &str,
        expr: &str,
        output_fields: Vec<&str>,
        partition_names: Vec<&str>,
        travel_timestamp: u64,
        guarantee_timestamp: u64,
        query_params: Option<HashMap<String, String>>,
    ) -> Result<QueryResult> {
        let request = proto::milvus::QueryRequest {
            base: Some(new_msg(MsgType::Retrieve)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            expr: expr.to_string(),
            output_fields: output_fields.into_iter().map(|s| s.to_string()).collect(),
            partition_names: partition_names.into_iter().map(|s| s.to_string()).collect(),
            travel_timestamp,
            guarantee_timestamp,
            query_params: query_params
                .map(|x| {
                    x.into_iter()
                        .map(|(k, v)| proto::common::KeyValuePair {
                            key: k.clone(),
                            value: v.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        };

        let response = self.client.clone().query(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = QueryResult {
            fields_data: response.fields_data.into_iter().map(|x| x.into()).collect(),
            collection_name: response.collection_name,
        };

        Ok(res)
    }

    pub async fn get_flush_state(&self, segment_ids: Vec<i64>) -> Result<bool> {
        let request = proto::milvus::GetFlushStateRequest {
            segment_i_ds: segment_ids,
        };

        let response = self
            .client
            .clone()
            .get_flush_state(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.flushed)
    }

    pub async fn get_persistent_segment_info(
        &self,
        db_name: &str,
        collection_name: &str,
    ) -> Result<Vec<PersistentSegmentInfo>> {
        let request = proto::milvus::GetPersistentSegmentInfoRequest {
            base: Some(new_msg(MsgType::ShowSegments)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_persistent_segment_info(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = response
            .infos
            .into_iter()
            .map(|x| PersistentSegmentInfo {
                segment_id: x.segment_id,
                collection_id: x.collection_id,
                partition_id: x.partition_id,
                num_rows: x.num_rows,
                state: SegmentState::from_i32(x.state).unwrap(),
            })
            .collect();

        Ok(res)
    }

    pub async fn get_query_segment_info(
        &self,
        db_name: &str,
        collection_name: &str,
    ) -> Result<Vec<QuerySegmentInfo>> {
        let request = proto::milvus::GetQuerySegmentInfoRequest {
            base: Some(new_msg(MsgType::SegmentInfo)),
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
        };

        let response = self
            .client
            .clone()
            .get_query_segment_info(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = response
            .infos
            .into_iter()
            .map(|x| QuerySegmentInfo {
                segment_id: x.segment_id,
                collection_id: x.collection_id,
                partition_id: x.partition_id,
                mem_size: x.mem_size,
                num_rows: x.num_rows,
                index_name: x.index_name,
                index_id: x.index_id,
                node_id: x.node_id,
                state: SegmentState::from_i32(x.state).unwrap(),
                node_ids: x.node_ids,
            })
            .collect();

        Ok(res)
    }

    pub async fn get_replicas(
        &self,
        collection_id: i64,
        with_shard_nodes: bool,
    ) -> Result<Vec<ReplicaInfo>> {
        let request = proto::milvus::GetReplicasRequest {
            base: Some(new_msg(MsgType::GetReplicas)),
            collection_id,
            with_shard_nodes,
        };

        let response = self
            .client
            .clone()
            .get_replicas(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = response.replicas.into_iter().map(|x| x.into()).collect();

        Ok(res)
    }

    pub async fn dummy(&self, request_type: &str) -> Result<String> {
        let request = proto::milvus::DummyRequest {
            request_type: request_type.to_string(),
        };

        let response = self.client.clone().dummy(request).await?.into_inner();

        Ok(response.response)
    }

    pub async fn register_link(&self) -> Result<Address> {
        let request = proto::milvus::RegisterLinkRequest {};

        let response = self
            .client
            .clone()
            .register_link(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.address.unwrap_or_default().into())
    }

    /// `request` is of jsonic format
    pub async fn get_metrics(&self, request: String) -> Result<Metrics> {
        let request = proto::milvus::GetMetricsRequest {
            request,
            ..Default::default()
        };

        let response = self.client.clone().get_metrics(request).await?.into_inner();

        status_to_result(&response.status)?;

        Ok(Metrics {
            response: response.response,
            component_name: response.component_name,
        })
    }

    pub async fn get_component_states(&self) -> Result<ComponentState> {
        let request = proto::milvus::GetComponentStatesRequest {};

        let response = self
            .client
            .clone()
            .get_component_states(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = ComponentState {
            state: response.state.map(|x| x.into()),
            subcomponent_states: response
                .subcomponent_states
                .into_iter()
                .map(|x| x.into())
                .collect(),
        };

        Ok(res)
    }

    pub async fn load_balance(
        &self,
        src_node_id: i64,
        dst_node_ids: Vec<i64>,
        sealed_segment_ids: Vec<i64>,
        collection_name: &str,
    ) -> Result<()> {
        let request = proto::milvus::LoadBalanceRequest {
            base: Some(new_msg(MsgType::LoadBalanceSegments)),
            collection_name: collection_name.to_string(),
            src_node_id,
            dst_node_i_ds: dst_node_ids,
            sealed_segment_i_ds: sealed_segment_ids,
        };

        let response = self
            .client
            .clone()
            .load_balance(request)
            .await?
            .into_inner();

        status_to_result(&Some(response))?;

        Ok(())
    }

    pub async fn get_compaction_state(&self, compaction_id: i64) -> Result<CompactionStateResult> {
        let request = proto::milvus::GetCompactionStateRequest { compaction_id };

        let response = self
            .client
            .clone()
            .get_compaction_state(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = CompactionStateResult {
            state: CompactionState::from_i32(response.state).unwrap(),
            executing_plan_no: response.executing_plan_no,
            timeout_plan_no: response.timeout_plan_no,
            completed_plan_no: response.completed_plan_no,
            failed_plan_no: response.failed_plan_no,
        };

        Ok(res)
    }

    pub async fn manual_compaction(&self, collection_id: i64, time_travel: u64) -> Result<i64> {
        let request = proto::milvus::ManualCompactionRequest {
            collection_id,
            timetravel: time_travel,
        };

        let response = self
            .client
            .clone()
            .manual_compaction(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.compaction_id)
    }

    pub async fn get_compaction_state_with_plans(
        &self,
        compaction_id: i64,
    ) -> Result<CompactionPlan> {
        let request = proto::milvus::GetCompactionPlansRequest { compaction_id };

        let response = self
            .client
            .clone()
            .get_compaction_state_with_plans(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = CompactionPlan {
            state: CompactionState::from_i32(response.state).unwrap(),
            merge_infos: response
                .merge_infos
                .into_iter()
                .map(|x| CompactionMergeInfo {
                    sources: x.sources,
                    target: x.target,
                })
                .collect(),
        };

        Ok(res)
    }

    pub async fn import(
        &self,
        collection_name: &str,
        partition_name: &str,
        channel_names: Vec<&str>,
        row_based: bool,
        files: Vec<&str>,
        options: HashMap<String, String>,
    ) -> Result<Vec<i64>> {
        let request = proto::milvus::ImportRequest {
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            channel_names: channel_names.iter().map(|x| x.to_string()).collect(),
            row_based,
            files: files.iter().map(|x| x.to_string()).collect(),
            options: options
                .into_iter()
                .map(|(key, value)| proto::common::KeyValuePair { key, value })
                .collect(),
        };

        let response = self.client.clone().import(request).await?.into_inner();

        status_to_result(&response.status)?;

        Ok(response.tasks)
    }

    pub async fn get_import_state(&self, task_id: i64) -> Result<ImportStateResult> {
        let request = proto::milvus::GetImportStateRequest { task: task_id };

        let response = self
            .client
            .clone()
            .get_import_state(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = ImportStateResult {
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
        };

        Ok(res)
    }

    /// List the tasks of the target collection.
    /// # Arguments
    ///
    /// * `collection_name` - The name of the target collection. If `collection_name` is empty, all tasks will be returned.
    ///
    /// * `limit` - The maximum number of tasks to return. If limit is 0, all tasks will be returned.
    pub async fn list_import_tasks(
        &self,
        collection_name: &str,
        limit: i64,
    ) -> Result<Vec<ImportStateResult>> {
        let request = proto::milvus::ListImportTasksRequest {
            collection_name: collection_name.to_string(),
            limit,
        };

        let response = self
            .client
            .clone()
            .list_import_tasks(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = response.tasks.into_iter().map(|task| task.into()).collect();

        Ok(res)
    }

    /// Create a credential for the user.
    ///
    /// # Arguments
    ///
    /// * `username` - The name of the user.
    ///
    /// * `password` - The ciphertext password of the user.
    ///
    /// * `created_utc_timestamps` - The created time.
    ///
    /// * `modified_utc_timestamps` - The modified time.
    pub async fn create_credential(
        &self,
        username: &str,
        password: &str,
        created_utc_timestamps: u64,
        modified_utc_timestamps: u64,
    ) -> Result<()> {
        let request = proto::milvus::CreateCredentialRequest {
            base: Some(new_msg(MsgType::CreateCredential)),
            username: username.to_string(),
            password: password.to_string(),
            created_utc_timestamps,
            modified_utc_timestamps,
        };

        let status = self
            .client
            .clone()
            .create_credential(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Update the password of the user.
    ///
    /// # Arguments
    ///
    /// * `username` - The name of the user.
    ///
    /// * `old_password` - The old password of the user.
    ///
    /// * `new_password` - The new password of the user.
    ///
    /// * `created_utc_timestamps` - The created time.
    ///
    /// * `modified_utc_timestamps` - The modified time.
    pub async fn update_credential(
        &self,
        username: &str,
        old_password: &str,
        new_password: &str,
        created_utc_timestamps: u64,
        modified_utc_timestamps: u64,
    ) -> Result<()> {
        let request = proto::milvus::UpdateCredentialRequest {
            base: Some(new_msg(MsgType::UpdateCredential)),
            username: username.to_string(),
            old_password: old_password.to_string(),
            new_password: new_password.to_string(),
            created_utc_timestamps,
            modified_utc_timestamps,
        };

        let status = self
            .client
            .clone()
            .update_credential(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    /// Delete the credential of the user.
    ///
    /// # Arguments
    ///
    /// * `username` - The name of the user.
    pub async fn delete_credential(&self, username: &str) -> Result<()> {
        let request = proto::milvus::DeleteCredentialRequest {
            base: Some(new_msg(MsgType::DeleteCredential)),
            username: username.to_string(),
        };

        let status = self
            .client
            .clone()
            .delete_credential(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn list_credential_usernames(&self) -> Result<Vec<String>> {
        let request = proto::milvus::ListCredUsersRequest {
            base: Some(new_msg(MsgType::ListCredUsernames)),
        };

        let response = self
            .client
            .clone()
            .list_cred_users(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(response.usernames)
    }

    pub async fn create_role(&self, role: Option<RoleEntity>) -> Result<()> {
        let request = proto::milvus::CreateRoleRequest {
            base: Some(new_msg(MsgType::CreateRole)),
            entity: role.map(|x| x.into()),
        };

        let status = self.client.clone().create_role(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn drop_role(&self, role_name: &str) -> Result<()> {
        let request = proto::milvus::DropRoleRequest {
            base: Some(new_msg(MsgType::DropRole)),
            role_name: role_name.to_string(),
        };

        let status = self.client.clone().drop_role(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn operate_user_role(
        &self,
        username: &str,
        role_name: &str,
        ty: OperateUserRoleType,
    ) -> Result<()> {
        let request = proto::milvus::OperateUserRoleRequest {
            base: Some(new_msg(MsgType::OperateUserRole)),
            username: username.to_string(),
            role_name: role_name.to_string(),
            r#type: ty as i32,
        };

        let status = self
            .client
            .clone()
            .operate_user_role(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn select_role(
        &self,
        role: Option<RoleEntity>,
        include_user_info: bool,
    ) -> Result<Vec<RoleResult>> {
        let request = proto::milvus::SelectRoleRequest {
            base: Some(new_msg(MsgType::SelectRole)),
            role: role.map(|role| role.into()),
            include_user_info,
        };

        let response = self.client.clone().select_role(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = response
            .results
            .into_iter()
            .map(|role| RoleResult {
                role: role.role.map(|role| role.into()),
                users: role.users.into_iter().map(|user| user.into()).collect(),
            })
            .collect();

        Ok(res)
    }

    pub async fn select_user(
        &self,
        user: Option<UserEntity>,
        include_role_info: bool,
    ) -> Result<Vec<User>> {
        let request = proto::milvus::SelectUserRequest {
            base: Some(new_msg(MsgType::SelectUser)),
            user: user.map(|user| user.into()),
            include_role_info,
        };

        let response = self.client.clone().select_user(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = response
            .results
            .into_iter()
            .map(|user| User {
                user: user.user.map(|user| user.into()),
                roles: user.roles.into_iter().map(|role| role.into()).collect(),
            })
            .collect();

        Ok(res)
    }

    pub async fn operate_privilege(
        &self,
        entity: GrantEntity,
        ty: OperatePrivilegeType,
    ) -> Result<()> {
        let request = proto::milvus::OperatePrivilegeRequest {
            base: Some(new_msg(MsgType::OperatePrivilege)),
            entity: Some(entity.into()),
            r#type: ty as i32,
        };

        let status = self
            .client
            .clone()
            .operate_privilege(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn select_grant(&self, object_name: &str) -> Result<Vec<GrantEntity>> {
        let entity = GrantEntity {
            object_name: object_name.to_string(),
            ..Default::default()
        };
        let request = proto::milvus::SelectGrantRequest {
            base: Some(new_msg(MsgType::SelectGrant)),
            entity: Some(entity.into()),
        };

        let response = self
            .client
            .clone()
            .select_grant(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        let res = response
            .entities
            .into_iter()
            .map(|grant| GrantEntity {
                role: grant.role.map(|x| x.into()),
                object: grant.object.map(|x| x.into()),
                object_name: grant.object_name,
                grantor: grant.grantor.map(|x| x.into()),
            })
            .collect();

        Ok(res)
    }

    pub async fn get_version(&self) -> Result<String> {
        let request = proto::milvus::GetVersionRequest {};

        let response = self.client.clone().get_version(request).await?.into_inner();

        status_to_result(&response.status)?;

        Ok(response.version)
    }

    pub async fn check_health(&self) -> Result<Health> {
        let request = proto::milvus::CheckHealthRequest {};

        let response = self
            .client
            .clone()
            .check_health(request)
            .await?
            .into_inner();

        status_to_result(&response.status)?;

        Ok(Health {
            is_healthy: response.is_healthy,
            reasons: response.reasons,
        })
    }
}

#[derive(Clone)]
pub struct AuthInterceptor {
    token: Option<String>,
}

impl Interceptor for AuthInterceptor {
    fn call(
        &mut self,
        mut req: Request<()>,
    ) -> std::result::Result<tonic::Request<()>, tonic::Status> {
        if let Some(ref token) = self.token {
            let header_value = format!("{}", token);
            req.metadata_mut()
                .insert("authorization", header_value.parse().unwrap());
        }

        Ok(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{CollectionSchema, FieldSchema, FieldType};

    fn get_vdb_host_address() -> String {
        std::env::var("VDB_HOST").expect("VDB_HOST is not set")
    }

    #[tokio::test]
    async fn test_client_new() {
        let result = Client::new(
            get_vdb_host_address().as_str(),
            19530,
            None,
            None,
            Some(std::time::Duration::from_secs(10)),
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_client_collection() -> Result<()> {
        let client = Client::new(
            get_vdb_host_address().as_str(),
            19530,
            None,
            None,
            Some(std::time::Duration::from_secs(10)),
        )
        .await?;

        // create a collection `c1`

        let c1_name = "c1";
        let c1_schema = CollectionSchema::new(
            "c1",
            vec![FieldSchema::new(
                "field1",
                FieldType::Int64(true, true),
                Some("This is the first field of `c1` collection"),
            )],
            Some("This is `c1` collection"),
        );
        let result = client
            .create_collection(c1_name, c1_schema, None, None, None)
            .await;
        assert!(result.is_ok());

        // create a collection `c2`
        let c2_name = "c2";
        let c2_schema = CollectionSchema::new(
            "c2",
            vec![FieldSchema::new(
                "field1",
                FieldType::VarChar(20, true, false),
                Some("This is the first field of `c2` collection"),
            )],
            Some("This is `c2` collection"),
        );
        let result = client
            .create_collection(c2_name, c2_schema, None, None, None)
            .await;
        assert!(result.is_ok());

        // has collection
        assert!(client.has_collection(c1_name, None).await?);
        assert!(client.has_collection(c2_name, None).await?);

        // list collections
        let collection_info_vec = client.show_collections(vec![c1_name, c2_name]).await?;
        assert_eq!(collection_info_vec.len(), 2);
        assert!([c1_name, c2_name].contains(&collection_info_vec[0].name.as_str()));
        assert!([c1_name, c2_name].contains(&collection_info_vec[1].name.as_str()));

        // get collection
        let c1_metadata = client.describe_collection(c1_name, None).await?;
        assert_eq!(c1_metadata.name, c1_name);

        // drop the `c1` collection
        let result = client.drop_collection(c1_name).await;
        assert!(result.is_ok());

        assert!(!client.has_collection(c1_name, None).await?);

        Ok(())
    }
}
