use base64::engine::general_purpose;
use base64::Engine;
use milvus::proto::common::{ConsistencyLevel, KeyValuePair, MsgType};
use milvus::proto::milvus::milvus_service_client::MilvusServiceClient;
use num_traits::FromPrimitive;
use prost::{bytes::BytesMut, Message};
use tonic::codegen::InterceptedService;
use tonic::service::Interceptor;
use tonic::transport::Channel;
use tonic::Request;

use crate::{
    my_collection::{
        CollectionInfo, CollectionMetadata, FieldData, FlushResult, IndexInfo, IndexProgress,
        IndexState, MutationResult, PartitionInfo, PersistentSegmentInfo, QueryResult,
        QuerySegmentInfo, SearchResult, SegmentState,
    },
    my_error::{Error, Result},
    schema::CollectionSchema,
    utils::{new_msg, status_to_result},
};

use std::collections::HashMap;

#[derive(Debug)]
pub struct Client {
    client: MilvusServiceClient<InterceptedService<Channel, AuthInterceptor>>,
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

        let client = MilvusServiceClient::with_interceptor(conn, auth_interceptor);

        Ok(Self { client })
    }

    pub async fn create_collection(
        &self,
        schema: CollectionSchema,
        shards_num: Option<i32>,
        level: Option<ConsistencyLevel>,
    ) -> Result<()> {
        let shards_num = shards_num.unwrap_or(2);

        let consistency_level = level.unwrap_or(ConsistencyLevel::Bounded);

        let schema: milvus::proto::schema::CollectionSchema = schema.into();
        let mut buf = BytesMut::new();
        schema.encode(&mut buf)?;

        let request = milvus::proto::milvus::CreateCollectionRequest {
            base: Some(new_msg(MsgType::CreateCollection)),
            collection_name: schema.name.to_string(),
            schema: buf.to_vec(),
            shards_num,
            consistency_level: consistency_level.into(),
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

    pub async fn drop_collection(&self, name: &str) -> Result<()> {
        let request = milvus::proto::milvus::DropCollectionRequest {
            base: Some(new_msg(MsgType::DropCollection)),
            collection_name: name.to_string(),
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

    pub async fn has_collection(&self, name: &str) -> Result<bool> {
        let request = milvus::proto::milvus::HasCollectionRequest {
            base: Some(new_msg(MsgType::HasCollection)),
            collection_name: name.to_string(),
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

    ///
    /// # Arguments
    ///
    /// * `name` - collection name
    ///
    /// * `replica_num` - replica number to load, default by 1
    ///
    pub async fn load_collection(&self, name: &str, replica_num: Option<i32>) -> Result<()> {
        let replica_number = replica_num.unwrap_or(1);

        let request = milvus::proto::milvus::LoadCollectionRequest {
            base: Some(new_msg(MsgType::LoadCollection)),
            collection_name: name.to_string(),
            replica_number,
            ..Default::default()
        };

        let status = self
            .client
            .clone()
            .load_collection(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn release_collection(&self, name: &str) -> Result<()> {
        let request = milvus::proto::milvus::ReleaseCollectionRequest {
            base: Some(new_msg(MsgType::ReleaseCollection)),
            collection_name: name.to_string(),
            ..Default::default()
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
    pub async fn describe_collection(&self, name: &str) -> Result<CollectionMetadata> {
        let request = milvus::proto::milvus::DescribeCollectionRequest {
            base: Some(new_msg(MsgType::DescribeCollection)),
            collection_name: name.to_string(),
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
    pub async fn get_collection_stats(&self, name: &str) -> Result<HashMap<String, String>> {
        let request = milvus::proto::milvus::GetCollectionStatisticsRequest {
            base: Some(new_msg(MsgType::GetCollectionStatistics)),
            collection_name: name.to_string(),
            ..Default::default()
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
    pub async fn show_collections(&self) -> Result<Vec<CollectionInfo>> {
        let request = milvus::proto::milvus::ShowCollectionsRequest {
            base: Some(new_msg(MsgType::ShowCollections)),
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
            println!("collection_names: {}", response.collection_names[i]);
            info_vec.push(CollectionInfo {
                name: response.collection_names[i].clone(),
                id: response.collection_ids[i],
                created_timestamp: response.created_timestamps[i],
                created_utc_timestamp: response.created_utc_timestamps[i],
                in_memory_percentage: response.in_memory_percentages[i],
                query_service_available: response.query_service_available[i],
            });
        }

        Ok(info_vec)
    }

    /// Alter collection.
    pub async fn alter_collection(
        &self,
        name: &str,
        properties: Vec<(String, String)>,
    ) -> Result<()> {
        let request = milvus::proto::milvus::AlterCollectionRequest {
            base: Some(new_msg(MsgType::AlterCollection)),
            collection_name: name.to_string(),
            properties: properties
                .into_iter()
                .map(|(key, value)| milvus::proto::common::KeyValuePair { key, value })
                .collect(),
            ..Default::default()
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
        collection_name: &str,
        partition_name: &str,
    ) -> Result<()> {
        let request = milvus::proto::milvus::CreatePartitionRequest {
            base: Some(new_msg(MsgType::CreatePartition)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            ..Default::default()
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
    pub async fn drop_partition(&self, collection_name: &str, partition_name: &str) -> Result<()> {
        let request = milvus::proto::milvus::DropPartitionRequest {
            base: Some(new_msg(MsgType::DropPartition)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            ..Default::default()
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
    pub async fn has_partition(&self, collection_name: &str, partition_name: &str) -> Result<bool> {
        let request = milvus::proto::milvus::HasPartitionRequest {
            base: Some(new_msg(MsgType::HasPartition)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        partition_names: Vec<&str>,
        replica_number: i32,
    ) -> Result<()> {
        let request = milvus::proto::milvus::LoadPartitionsRequest {
            base: Some(new_msg(MsgType::LoadPartitions)),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.iter().map(|x| x.to_string()).collect(),
            replica_number,
            ..Default::default()
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
        collection_name: &str,
        partition_names: Vec<&str>,
    ) -> Result<()> {
        let request = milvus::proto::milvus::ReleasePartitionsRequest {
            base: Some(new_msg(MsgType::ReleasePartitions)),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.iter().map(|x| x.to_string()).collect(),
            ..Default::default()
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
        collection_name: &str,
        partition_name: &str,
    ) -> Result<HashMap<String, String>> {
        let request = milvus::proto::milvus::GetPartitionStatisticsRequest {
            base: Some(new_msg(MsgType::GetPartitionStatistics)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        partition_names: Option<Vec<&str>>,
    ) -> Result<Vec<PartitionInfo>> {
        let request = milvus::proto::milvus::ShowPartitionsRequest {
            base: Some(new_msg(MsgType::ShowPartitions)),
            collection_name: collection_name.to_string(),
            partition_names: partition_names
                .unwrap_or_default()
                .iter()
                .map(|x| x.to_string())
                .collect(),
            ..Default::default()
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

    pub async fn create_alias(&self, collection_name: &str, alias: &str) -> Result<()> {
        let request = milvus::proto::milvus::CreateAliasRequest {
            base: Some(new_msg(MsgType::CreateAlias)),
            collection_name: collection_name.to_string(),
            alias: alias.to_string(),
            ..Default::default()
        };

        let status = self
            .client
            .clone()
            .create_alias(request)
            .await?
            .into_inner();

        status_to_result(&Some(status))
    }

    pub async fn drop_alias(&self, alias: &str) -> Result<()> {
        let request = milvus::proto::milvus::DropAliasRequest {
            base: Some(new_msg(MsgType::DropAlias)),
            alias: alias.to_string(),
            ..Default::default()
        };

        let status = self.client.clone().drop_alias(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn alter_alias(&self, alias: &str, collection_name: &str) -> Result<()> {
        let request = milvus::proto::milvus::AlterAliasRequest {
            base: Some(new_msg(MsgType::AlterAlias)),
            alias: alias.to_string(),
            collection_name: collection_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        field_name: &str,
        extra_params: Option<HashMap<String, String>>,
        index_name: &str,
    ) -> Result<()> {
        let request = milvus::proto::milvus::CreateIndexRequest {
            base: Some(new_msg(MsgType::CreateIndex)),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            extra_params: extra_params
                .unwrap_or_default()
                .iter()
                .map(|(key, value)| KeyValuePair {
                    key: key.clone(),
                    value: value.clone(),
                })
                .collect(),
            index_name: index_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<Vec<IndexInfo>> {
        let request = milvus::proto::milvus::DescribeIndexRequest {
            base: Some(new_msg(MsgType::DescribeIndex)),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<IndexState> {
        let request = milvus::proto::milvus::GetIndexStateRequest {
            base: Some(new_msg(MsgType::GetIndexState)),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<IndexProgress> {
        let request = milvus::proto::milvus::GetIndexBuildProgressRequest {
            base: Some(new_msg(MsgType::GetIndexBuildProgress)),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
            ..Default::default()
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
        collection_name: &str,
        field_name: &str,
        index_name: &str,
    ) -> Result<()> {
        let request = milvus::proto::milvus::DropIndexRequest {
            base: Some(new_msg(MsgType::DropIndex)),
            collection_name: collection_name.to_string(),
            field_name: field_name.to_string(),
            index_name: index_name.to_string(),
            ..Default::default()
        };

        let status = self.client.clone().drop_index(request).await?.into_inner();

        status_to_result(&Some(status))
    }

    pub async fn insert(
        &self,
        collection_name: &str,
        partition_name: &str,
        fields_data: Vec<FieldData>,
    ) -> Result<MutationResult> {
        let request = milvus::proto::milvus::InsertRequest {
            base: Some(new_msg(MsgType::Insert)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            num_rows: fields_data.len() as u32,
            fields_data: fields_data
                .into_iter()
                .map(|field_data| field_data.into())
                .collect(),
            ..Default::default()
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
        collection_name: &str,
        partition_name: &str,
        expr: &str,
    ) -> Result<MutationResult> {
        let request = milvus::proto::milvus::DeleteRequest {
            base: Some(new_msg(MsgType::Delete)),
            collection_name: collection_name.to_string(),
            partition_name: partition_name.to_string(),
            expr: expr.to_string(),
            ..Default::default()
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
        collection_name: &str,
        partition_names: Vec<&str>,
        dsl: &str,
        placeholder_group: Vec<u8>,
        dsl_type: i32,
        output_fields: Vec<String>,
        search_params: HashMap<String, String>,
        travel_timestamp: u64,
        guarantee_timestamp: u64,
        nq: i64,
    ) -> Result<SearchResult> {
        let request = milvus::proto::milvus::SearchRequest {
            base: Some(new_msg(MsgType::Search)),
            collection_name: collection_name.to_string(),
            partition_names: partition_names.into_iter().map(|s| s.to_string()).collect(),
            dsl: dsl.to_string(),
            placeholder_group,
            dsl_type,
            output_fields,
            search_params: search_params
                .into_iter()
                .map(|(k, v)| KeyValuePair {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
            travel_timestamp,
            guarantee_timestamp,
            nq,
            ..Default::default()
        };

        let response = self.client.clone().search(request).await?.into_inner();

        status_to_result(&response.status)?;

        let res = SearchResult {
            results: response.results.map(|x| x.into()),
            collection_name: response.collection_name,
        };

        Ok(res)
    }

    pub async fn flush(&self, collection_names: Vec<&str>) -> Result<FlushResult> {
        let request = milvus::proto::milvus::FlushRequest {
            base: Some(new_msg(MsgType::Flush)),
            collection_names: collection_names
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
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
        collection_name: &str,
        expr: &str,
        output_fields: Vec<&str>,
        partition_names: Vec<&str>,
        travel_timestamp: u64,
        guarantee_timestamp: u64,
        query_params: Option<HashMap<String, String>>,
    ) -> Result<QueryResult> {
        let request = milvus::proto::milvus::QueryRequest {
            base: Some(new_msg(MsgType::Retrieve)),
            collection_name: collection_name.to_string(),
            expr: expr.to_string(),
            output_fields: output_fields.into_iter().map(|s| s.to_string()).collect(),
            partition_names: partition_names.into_iter().map(|s| s.to_string()).collect(),
            travel_timestamp,
            guarantee_timestamp,
            query_params: query_params
                .map(|x| {
                    x.into_iter()
                        .map(|(k, v)| KeyValuePair {
                            key: k.clone(),
                            value: v.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            ..Default::default()
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
        let request = milvus::proto::milvus::GetFlushStateRequest {
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
        let request = milvus::proto::milvus::GetPersistentSegmentInfoRequest {
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
        let request = milvus::proto::milvus::GetQuerySegmentInfoRequest {
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

    fn get_vdb_host_address() -> String {
        std::env::var("VDB_HOST").expect("VDB_HOST is not set")
    }

    #[tokio::test]
    async fn test_new_client_new() {
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

    // #[tokio::test]
    // async fn test_client_collection() -> VDBResult<()> {
    //     let mut client = Client::new(
    //         get_vdb_host_address().as_str(),
    //         19530,
    //         None,
    //         None,
    //         Some(std::time::Duration::from_secs(10)),
    //     )
    //     .await?;

    //     // create a collection `c1`
    //     let c1_name = "c1";
    //     let c1_schema = CollectionSchema::new(
    //         c1_name,
    //         vec![FieldSchema::new(
    //             "field1",
    //             FieldType::Int64(true, true),
    //             Some("This is the first field of `c1` collection"),
    //         )],
    //         Some("This is `c1` collection"),
    //     )?;
    //     let c1_options = CreateCollectionOptions::default();
    //     let result = client.create_collection(c1_schema, Some(c1_options)).await;
    //     assert!(result.is_ok());

    //     // create a collection `c2`
    //     let c2_name = "c2";
    //     let c2_schema = CollectionSchema::new(
    //         c2_name,
    //         vec![FieldSchema::new(
    //             "field1",
    //             FieldType::VarChar(20, true, false),
    //             Some("This is the first field of `c2` collection"),
    //         )],
    //         Some("This is `c2` collection"),
    //     )?;
    //     let c2_options = CreateCollectionOptions::default();
    //     let result = client.create_collection(c2_schema, Some(c2_options)).await;
    //     assert!(result.is_ok());

    //     // has collection
    //     assert!(client.has_collection(c1_name));
    //     assert!(client.has_collection(c2_name));

    //     // list collections
    //     let names = client.collection_names();
    //     assert_eq!(names.len(), 2);
    //     assert!(names.contains(&c1_name));
    //     assert!(names.contains(&c2_name));

    //     // get collection
    //     let c1 = client.collection(c1_name);
    //     assert!(c1.is_some());

    //     // drop the `c1` collection
    //     let result = client.remove_collection(c1_name).await;
    //     assert!(result.is_ok());

    //     assert!(!client.has_collection(c1_name));

    //     Ok(())
    // }
}
