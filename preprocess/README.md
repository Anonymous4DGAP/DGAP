1. Download CoreNLP source file from url:
   
   [Overview - CoreNLP (stanfordnlp.github.io)](https://stanfordnlp.github.io/CoreNLP/)

2. Unzip the source file.

3. Start CoreNLP service (taking version 4.4.0 as an example)：
   
   `cd stanford-corenlp-4.4.0`
   
   `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 50000`

4. Call the service as follows:
   
   `nlp = StanfordCoreNLP('http://localhost', port=9000)`
5. Modify the name of the dataset for preprocessing in `config.py`.
6. Run the files in the following order:
   
   `filter.py` to filter useless information and cut words;
   
   `parser.py` to perform dependency syntactic analysis on texts;
   
   `constructor.py` to build dual graph of texts.