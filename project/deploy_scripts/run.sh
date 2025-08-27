while true; do
    [ -e stopme ] && break
    streamlit run main.py
done