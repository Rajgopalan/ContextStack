"""Machine-readable Agent Operation Procedure schema."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AOPField(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


class AOPStep(BaseModel):
    id: str = Field(description="Stable step id, e.g. step_1")
    name: str
    description: str = ""
    action_type: str = Field(
        description="One of: data_extraction, api_call, decision, validation, manual, notification, file_operation"
    )
    tool: Optional[str] = Field(default=None, description="Tool/system to use, if any")
    parameters: Dict = Field(default_factory=dict)
    expected_output: str = ""
    validation_criteria: Optional[str] = None
    on_failure: Optional[str] = None


class AOP(BaseModel):
    id: str
    title: str
    version: str = "1.0.0"
    description: str = ""
    source_documents: List[str] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    inputs: List[AOPField] = Field(default_factory=list)
    outputs: List[AOPField] = Field(default_factory=list)
    steps: List[AOPStep] = Field(default_factory=list)
    error_handling: List[str] = Field(default_factory=list)
    notes: str = ""


AOP_JSON_SCHEMA = AOP.model_json_schema()
