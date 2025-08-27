while true; do
    [ -e stopme ] && break
    streamlit run main.py --server.port 8900
done