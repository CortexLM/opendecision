from opendecision import Decider
r = Decider().predict({"state": "Refund the duplicate charge today.", "questions": {
    "dept": {"type": "choice", "instructions": "Which department?", "criteria": {"billing": "money", "tech": "bugs"}}}})
print(r)
