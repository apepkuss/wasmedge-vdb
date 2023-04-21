use base64::engine::general_purpose;
use base64::Engine;
use milvus::proto::common::{ConsistencyLevel, MsgType};
use milvus::proto::milvus::milvus_service_client::MilvusServiceClient;
use num_traits::FromPrimitive;
use prost::{bytes::BytesMut, Message};
use tonic::codegen::InterceptedService;
use tonic::service::Interceptor;
use tonic::transport::Channel;
use tonic::Request;

use crate::{
    my_collection::{CollectionInfo, CollectionMetadata, PartitionInfo},
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
