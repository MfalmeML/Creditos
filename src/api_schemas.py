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
    credit_history: str
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str
    installment_commitment: int


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
