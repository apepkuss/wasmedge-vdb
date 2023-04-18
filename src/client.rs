use crate::{
    collection::{Collection, CollectionSchema},
    error::VDBResult,
    options::CreateCollectionOptions,
};

pub struct Client {
    inner: milvus::client::Client,
}
impl Client {
    pub async fn connect(url: String) -> VDBResult<Self> {
        match milvus::client::Client::new(url).await {
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
