from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os

class GoogleSheetsSettings(BaseSettings):
    OPEN_AI_KEY: str
    TABLE_TOKEN: str
    FILE_PATH: str

    scopes: List[str] = ['https://www.googleapis.com/auth/spreadsheets']

    model_config = SettingsConfigDict(env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env"), extra="ignore")

    def get_llm_key(self):
        return self.OPEN_AI_KEY
    
    def get_table_token(self):
        return self.TABLE_TOKEN
    
    def get_auth_tokens(self):
        return (self.FILE_PATH, self.scopes)

settings = GoogleSheetsSettings()
