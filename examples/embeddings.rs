use std::io::{self, Write};
use wasmedge_vdb_sdk::{
    client::Client,
    schema::{CollectionSchema, FieldSchema, FieldType},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!("[step-2] store the document embeddings in Milvus ... ");
    io::stdout().flush().unwrap();

    // create a vdb client
    let client = Client::new(
        get_vdb_host_address().as_str(),
        19530,
        None,
        None,
        Some(std::time::Duration::from_secs(10)),
    )
    .await?;

    // create a collection schema
    let collection_name = "flows_network_book";
    let collection_schema = CollectionSchema::new(
        collection_name,
        vec![
            FieldSchema::new(
                "book_id",
                FieldType::Int64(true, false),
                Some("This is `book_id` field"),
            ),
            FieldSchema::new(
                "book_name",
                FieldType::VarChar(200, false, false),
                Some("This is `book_name` field"),
            ),
            FieldSchema::new("book_intro", FieldType::FloatVector(1536), None),
        ],
        Some("A guide example for wasmedge-vdb-sdk"),
    );
    // Note: the if-statement is just for the example, you can remove it in your code
    if client.has_collection(collection_name, None).await? {
        client.drop_collection(collection_name).await?;
    }
    client
        .create_collection(collection_name, collection_schema, None, None, None)
        .await?;

    println!("Done");

    Ok(())
}

fn get_vdb_host_address() -> String {
    std::env::var("VDB_HOST").expect("VDB_HOST is not set")
}
