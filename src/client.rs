use crate::{
    collection::{Collection, CollectionSchema},
    error::VDBResult,
    options::CreateCollectionOptions,
};

pub struct Client {
    inner: milvus::client::Client,
}
impl Client {
    pub async fn new(
        host: &str,
        port: u16,
        username: Option<String>,
        password: Option<String>,
        timeout: std::time::Duration,
    ) -> VDBResult<Self> {
        let url = format!("{}:{}", host, port.to_string());

        match milvus::client::Client::with_timeout(url, timeout, username, password).await {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(Box::new(e.into())),
        }
    }

    pub async fn create_collection(
        &self,
        schema: CollectionSchema,
        options: Option<CreateCollectionOptions>,
    ) -> VDBResult<Collection> {
        match self
            .inner
            .create_collection(schema.into(), options.map(|x| x.into()))
            .await
        {
            Ok(inner) => Ok(Collection { inner }),
            Err(e) => Err(Box::new(e.into())),
        }
    }

    pub async fn list_collections(&self) -> VDBResult<Vec<String>> {
        match self.inner.list_collections().await {
            Ok(connections) => Ok(connections),
            Err(e) => Err(Box::new(e.into())),
        }
    }

    pub async fn get_collection(&self, name: &str) -> VDBResult<Collection> {
        match self.inner.get_collection(name).await {
            Ok(collection) => Ok(Collection { inner: collection }),
            Err(e) => Err(Box::new(e.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore]
    #[tokio::test]
    async fn test_client_new() {
        let result = Client::new(
            "http://127.0.0.1",
            19530,
            None,
            None,
            std::time::Duration::from_secs(10),
        )
        .await;
        assert!(result.is_ok());
    }
}
