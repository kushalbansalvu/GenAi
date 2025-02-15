import streamlit as st


st.title("Streamlit text input")

name=st.text_input("Enter your name:")

age = st.slider("Select your age:",0,100,125)
st.write("YOUR AGE IS:" , age)

options = ["php", "java", "golang", "c++"]
choice = st.selectbox("Ypur faviourite prog language is:", options)
st.write("You selected:", choice)

if name:
    st.write(f"Hello, {name}")