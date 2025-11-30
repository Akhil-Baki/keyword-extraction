import streamlit as st
import yake

def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

def main():
    st.title("Keyword Extraction Tool")
    st.write("Extract most meaningful keywords from your text using YAKE.")

    text = st.text_area("Enter your text here:")

    if st.button("Extract Keywords"):
        if text.strip():
            result = extract_keywords(text)
            st.subheader("Keywords:")
            for k in result:
                st.write("- " + k)
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
