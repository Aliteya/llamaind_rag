from .exel_tools import *
from .rag_tools import *
from .logging import logger

import asyncio

async def main():
    questions, ground_truth = load_questions("questions.xlsx")
    
    # vector_store = setup_qdrant("cache/embedding", "rag_cache")
    documents = load_json_db("rag_app/data/db.json")
    pipeline = create_ingestion_pipeline()

    nodes = await data_process(documents=documents, pipeline=pipeline)

    query_engine = setup_rag_pipeline(nodes)

    responses = await proccess_questions(query_engine, questions)

    # evaluated_responses = evaluate_responses(responses, ground_truth)
    
    data = prepare_data(responses)

    try:
        sheets_wrapper = GoogleSheetsWrapper()
        sheets_wrapper.get_google_sheets_service()

        sheet_name = "Aliteya"
        if not sheets_wrapper.sheet_exist(sheet_name):
            sheets_wrapper.create_sheet(sheet_name)

        sheets_wrapper.write_to_table(sheet_name, data)
    except Exception as e:
        logger.error(f"An error occurred while writing to Google Sheets: {e}")


if __name__ == '__main__':
    asyncio.run(main())