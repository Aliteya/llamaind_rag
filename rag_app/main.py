from .exel_tools import *
from .rag_tools import *
from .logging import logger

if __name__ == '__main__':
    questions = load_questions("questions.xlsx")

    query_engine = initialize_rag_pipeline("rag_app/data")

    responses = proccess_questions(query_engine, questions)
    i = 0
    for response in responses:
        i+=1
        print(i)
        # print(response)
        # print(f"Question: {response['question']}")
        # print(f"Answer: {response['answer']}")
        # print(f"Retrieved chunks: {response['retrieved_chunks']}")
        # print("-" * 20)
    
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