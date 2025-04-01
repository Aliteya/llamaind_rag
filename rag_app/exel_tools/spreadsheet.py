from ..core import settings
from ..logging import logger

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

class GoogleSheetsWrapper():
    def __init__(self):
        self.spreadsheet_id = settings.get_table_token()
        self.service = None

    def get_google_sheets_service(self):
        credentials = service_account.Credentials.from_service_account_file(
            settings.get_auth_tokens[0],
            scopes=settings.get_auth_tokens[1]
        )
        self.service = build('sheets', 'v4', credentials=credentials)

    def sheet_exist(self, sheet_name: str) -> bool:
        try:
            spreadsheet = self.service.spreadsheets().get(spreadsheetId=settings.get_table_token()).execute()
            sheets = spreadsheet.get("sheets", [])

            return any(sheet["properties"]["title"] == sheet_name for sheet in sheets)
        except HttpError as e:
            logger.error(f"Ошибка при проверке существования листа: {e}")
            return False

    def create_sheet(self, sheet_name: str):
        body = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_name
                        }
                    }
                }
            ]
        }
        try:
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=settings.get_table_token(),
                body=body
            ).execute()
            logger.info(f"Лист {sheet_name} успешно создан")
        except HttpError as e:
            logger.error(f"Ошибка при создании листа: {e}")

    def write_to_table(self, sheet_name: str, data: list):
        range_name = f"{sheet_name}!A1:E{len(data)}"
        body = {"values": data}
        try:
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                body=body
            ).execute()
            logger.info(f"Данные успешно записаны в лист '{sheet_name}'.")
        except HttpError as e:
            logger.error(f"Ошибка при записи данных: {e}")

def load_questions(file_path: str) -> list:
    dataframe = pd.read_excel(file_path)
    questions = dataframe.iloc[:, 0].dropna().tolist()
    return questions



