from pydantic import BaseModel
from typing import Optional

class Applicant(BaseModel):
    age: int
    credit_amount: float
    duration: int
    employment: str
    purpose: str
    checking_status: str
    savings_status: str

class Decision(BaseModel):
    approved: bool
    limit: Optional[float]
    rate: Optional[float]
    term: Optional[int]
    profit: Optional[float]
    pd: float
    lgd: float
    ead: float
    ecl: float
    fallback: str
    top_drivers: list
