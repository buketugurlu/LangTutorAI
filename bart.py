import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Model yolu
model_path = "C:/Users/Kullanıcı/Downloads/bart_model"

# Cihaz kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli ve tokenizer'ı yükleme
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = BartTokenizer.from_pretrained(model_path)

print("BART modeli ve tokenizer başarıyla yüklendi!")

# Cümle düzeltme fonksiyonu
def correct_sentence(sentence):
    input_text = sentence.strip()  # BART için yalnızca kullanıcı girdisini kullanıyoruz
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Streamlit başlık
st.title("Dilbilgisi Düzeltme Uygulaması (BART)")
st.write("Bir cümle girin, model düzeltmeleri yapacaktır.")

# Kullanıcıdan cümle alma
input_sentence = st.text_input("Cümle Girin", "")

# Düzeltme butonu
if st.button("Düzelt"):
    if input_sentence:
        corrected_sentence = correct_sentence(input_sentence)
        st.write("### Orijinal Cümle:")
        st.write(input_sentence)
        st.write("### Düzeltilmiş Cümle:")
        st.write(corrected_sentence)
    else:
        st.write("Lütfen bir cümle girin.")
