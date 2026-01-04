from pydantic import BaseModel

class FinancialData(BaseModel):
    """
    Pydantic model to validate the structure of the extracted financial data.
    Values are expected to be in millions of yen.
    """
    operating_cf: int
    investing_cf: int
    dividend_payout: int
