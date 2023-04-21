use crate::{
    collection::{Collection, CollectionSchema},
    error::VDBResult,
    options::CreateCollectionOptions,
};

pub struct Client {
    inner: milvus::client::Client,
    collections: std::collections::HashMap<String, Collection>,
}
impl Client {
    pub async fn new(
        host: &str,
        port: u16,
        username: Option<String>,
        password: Option<String>,
        timeout: Option<std::time::Duration>,
    ) -> VDBResult<Self> {
        let url = format!("{}:{}", host, port.to_string());
        let timeout = match timeout {
            Some(timeout) => timeout,
            None => std::time::Duration::from_secs(10),
        };

        match milvus::client::Client::with_timeout(url, timeout, username, password).await {
            Ok(inner) => {
                // sync up the existed collections
                let mut collections = std::collections::HashMap::new();
                if let Ok(collection_names) = inner.list_collections().await {
                    for name in collection_names {
                        if let Ok(inner_collection) = inner.get_collection(&name).await {
                            collections.insert(
                                name,
                                Collection {
                                    schema: inner_collection.schema().clone().into(),
                                    inner: inner_collection,
                                },
                            );
                        }
                    }
                }

                Ok(Self { inner, collections })
            }
            Err(e) => Err(Box::new(e.into())),
        }
    }

    pub async fn create_collection(
        &mut self,
        schema: CollectionSchema,
        options: Option<CreateCollectionOptions>,
    ) -> VDBResult<()> {
        match self
            .inner
            .create_collection(schema.clone().into(), options.map(|x| x.into()))
            .await
        {
            Ok(inner) => {
                self.collections
                    .insert(inner.name().to_string(), Collection { inner, schema });
                Ok(())
            }
            Err(e) => Err(Box::new(e.into())),
        }
    }

    pub fn collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    pub fn collection_names(&self) -> Vec<&str> {
        self.collections.keys().map(|x| x.as_str()).collect()
    }

    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }

    /// Release a collection from memory after a search or a query to reduce memory usage.
    pub async fn release_collection(&mut self, name: &str) -> VDBResult<()> {
        self.inner
            .release_collection(name)
            .await
            .map_err(|e| Box::new(e.into()))
    }

    /// Remove a collection and the data within
    pub async fn remove_collection(&mut self, name: &str) -> VDBResult<()> {
        self.collections.remove(name);
        self.inner
            .drop_collection(name)
            .await
            .map_err(|e| Box::new(e.into()))
    }

    /// Seal all entities in the current collection.
    ///
    /// Note that any insertion after a flush operation results in generating new segments. And only sealed segments can be indexed.
    pub async fn flush_collections(
        &self,
        names: &[&str],
    ) -> VDBResult<std::collections::HashMap<String, Vec<i64>>> {
        self.inner
            .flush_collections(names)
            .await
            .map_err(|e| Box::new(e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::{FieldSchema, FieldType};

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
    async fn test_client_collection() -> VDBResult<()> {
        let mut client = Client::new(
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
            c1_name,
            vec![FieldSchema::new(
                "field1",
                FieldType::Int64(true, true),
                Some("This is the first field of `c1` collection"),
            )],
            Some("This is `c1` collection"),
        )?;
        let c1_options = CreateCollectionOptions::default();
        let result = client.create_collection(c1_schema, Some(c1_options)).await;
        assert!(result.is_ok());

        // create a collection `c2`
        let c2_name = "c2";
        let c2_schema = CollectionSchema::new(
            c2_name,
            vec![FieldSchema::new(
                "field1",
                FieldType::VarChar(20, true, false),
                Some("This is the first field of `c2` collection"),
            )],
            Some("This is `c2` collection"),
        )?;
        let c2_options = CreateCollectionOptions::default();
        let result = client.create_collection(c2_schema, Some(c2_options)).await;
        assert!(result.is_ok());

        // has collection
        assert!(client.has_collection(c1_name));
        assert!(client.has_collection(c2_name));

        // list collections
        let names = client.collection_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&c1_name));
        assert!(names.contains(&c2_name));

        // get collection
        let c1 = client.collection(c1_name);
        assert!(c1.is_some());

        // drop the `c1` collection
        let result = client.remove_collection(c1_name).await;
        assert!(result.is_ok());

        assert!(!client.has_collection(c1_name));

        Ok(())
    }
}
