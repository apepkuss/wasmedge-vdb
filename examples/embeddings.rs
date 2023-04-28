use async_openai::{
    types::{CreateEmbeddingRequestArgs, Embedding},
    Client as AIClient,
};
use std::io::{self, Write};
use wasmedge_vdb_sdk::{
    client::Client,
    common::{DataType, Field, FieldData, ScalarField, VectorField},
    schema::{CollectionSchema, FieldSchema, FieldType},
    utils,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ======== step-1: create embeddings for a document with openai api ========
    print!("[step-1] create embeddings for a document with openai api ... ");
    io::stdout().flush().unwrap();

    let openai_client = AIClient::new();

    let book_contents = vec![
        "Why do programmers hate nature? It has too many bugs.",
        "Why was the computer cold? It left its Windows open.",
    ];
    let book_embeddings = gen_embeddings(&openai_client, book_contents).await?;

    println!("Done");

    // ========== step-2: create collection for storing data==========
    print!("[step-2] create collection ... ");
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

    // define a collection schema before creating collection.
    // the schema includes three fields: book_id, book_name, book_intro, which are defined with three different field schemas
    let collection_name = "flows_network_book";
    let collection_schema = CollectionSchema::new(
        collection_name,
        vec![
            FieldSchema::new(
                "book_id",
                FieldType::Int64(true, true),
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
    // create collection
    client
        .create_collection(collection_name, collection_schema, None, None, None)
        .await?;

    println!("Done");

    // ========== step-2: fill in the collection with data ==========
    print!("[step-2] Fill in the collection ... ");
    io::stdout().flush().unwrap();

    // psudo embeddings
    let embeddings = vec![0.1_f32, 0.2, 0.3];

    // insert data
    client
        .insert(
            collection_name,
            None,
            vec![
                FieldData::new(
                    "book_name",
                    DataType::VarChar,
                    Some(Field::Scalars(ScalarField::new(vec![
                        "The Hitchhiker's Guide to the Galaxy",
                    ]))),
                ),
                FieldData::new(
                    "book_intro",
                    DataType::FloatVector,
                    Some(Field::Vectors(VectorField::new(1536, embeddings))),
                ),
            ],
        )
        .await?;

    println!("Done");

    // ============== step-3: perform a vector query ==============
    print!("[step-3] perform a vector query ... ");
    io::stdout().flush().unwrap();

    // generate a vector from a user input text
    let text = vec!["the windows of the computer are open."];
    let embeddings = gen_embeddings(&openai_client, text).await?;
    let query_embedding = embeddings[0].embedding.clone();

    // load the newly created collection to memory before query
    client.load_collection(collection_name, 1).await?;

    // // perform vector similarity search
    // let data = vec![];
    // client.search(
    //     collection_name,
    //     vec![],
    //     "".to_string(),
    //     data,
    //     DslType::BoolExprV1,
    //     vec!["book_intro"],
    //     std::collections::HashMap::new(),
    //     0,
    //     utils::get_gts(),
    //     data.len() as i64,
    // );

    // let result = collection
    //     .search(
    //         vec![query_embedding.into()],
    //         "book_intro",
    //         1,
    //         MetricType::L2,
    //         vec!["book_name"],
    //         &SearchOption::default(),
    //     )
    //     .await?;

    // println!("Done");

    // // display the search result
    // println!("  *** search result: {:?}", result.len());
    // println!("  *** result[0].field: {:?}", result[0].field);

    Ok(())
}

fn get_vdb_host_address() -> String {
    std::env::var("VDB_HOST").expect("VDB_HOST is not set")
}

async fn gen_embeddings(
    client: &AIClient,
    text: Vec<&str>,
) -> Result<Vec<Embedding>, Box<dyn std::error::Error>> {
    let request = CreateEmbeddingRequestArgs::default()
        .model("text-embedding-ada-002")
        .input(text)
        .build()?;

    let response = client.embeddings().create(request).await?;

    Ok(response.data)
}
