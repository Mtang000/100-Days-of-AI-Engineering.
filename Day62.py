
from transformers import pipeline

ner_scanner = pipeline("ner", aggregation_strategy="simple")

messy_document = """
Hello, this is a reminder that Invoice #8842 from Amazon Web Services 
is due next week. Please ensure the payment is sent to our headquarters 
in Seattle, Washington. If you have any questions, contact Sarah Jenkins 
immediately.
"""

print(messy_document.strip())
print("-" * 30)

extracted_data = ner_scanner(messy_document)


for item in extracted_data:
    entity_group = item['entity_group']
    word = item['word']
    confidence = item['score'] * 100
    if entity_group == "PER":
        category = "Person Name"
    elif entity_group == "ORG":
        category = "Company/Org"
    elif entity_group == "LOC":
        category = "Location"
    else:
        category = "Other"

    print(f"[{category}] : {word:<20} (Confidence: {confidence:.1f}%)")
