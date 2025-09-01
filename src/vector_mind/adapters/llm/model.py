from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel


class StreamRequest(BaseModel):
    system_prompt: SystemMessage
    human_prompt: HumanMessage
