import spacy

# Load model NLP
nlp = spacy.load("en_core_web_sm")
caption = "In a beautiful day, a man is playing tennis with his friend!. There are trees and flowers around them. I see a dog running in the park."
doc = nlp(caption)

unique_objects = {}

for chunk in doc.noun_chunks:
    if chunk.root.pos_ == "PRON":
        continue
        
    root_lemma = chunk.root.lemma_.lower()
        
    chunk_text = chunk.text.lower().strip(" .,;?!")
    
    # Đề phòng chuỗi rỗng sau khi làm sạch
    if not chunk_text:
        continue
    
    # 4. Chống lặp: Giữ lại cụm mô tả dài nhất (nhiều chi tiết nhất)
    if root_lemma not in unique_objects:
        unique_objects[root_lemma] = chunk_text
    else:
        # Nếu cụm mới dài hơn cụm cũ đã lưu, thì ghi đè
        if len(chunk_text) > len(unique_objects[root_lemma]):
            unique_objects[root_lemma] = chunk_text

# Chuyển Dictionary value thành List cuối cùng
final_inventory = list(unique_objects.values())

print(final_inventory)