1. Download GloVe source file from url:
   
   `wget http://nlp.stanford.edu/data/glove.6B.zip`

2. Unzip the file `glove.6B.zip` in this path.
   
   `unzip glove.6B.zip "*.300d.txt"`

3. Run the code `handle_glove.py` in `/preprocess` to generate intermediate files `glove.6B.300d.npy` and `glove.6B.words.pkl`:
   
   `cd ../preprocess`
   
   `python handle_glove.py`