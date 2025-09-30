import guardrails as gd
rail_str = """
<rail version="0.1">
<script language ="python">
from guardrails.validators import Validator,EventDetail, register_validator
from typing import Dict,List
from profanity_check import predict
@register_validator(name="is-profanity-free",data_type="string")
class IsProfanityFree(Validator):
    global predict
    global EventDetail
    def validate(self,key,value,schema) -> Dict:
        text=value
        prediction = predict([value])
        if prediction[0] == 1:
            raise EventDetail(
            key,
            value,
            schema,
            f"Value {value} contains profanity language",
            "Sorry, I cannot respond as it contains profanity language",
        )
        return schema
</script>
<output>
    <string
    name="RAG application response"
    description="Question answering task based on context"
    format="is-profanity-free"
    on-fail-is-profanity-free="fix"
    />
</output>
<prompt>
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
    Context: {{context}}
    
</prompt>
</rail>
"""
guard=gd.Guard.for_rail_string(rail_str)