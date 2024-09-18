# chapter 2
| File | Description |
| ----------- | ----------- |
| [example_1.py](example_1.py) | Highlights the main components of the "facebook/opt-1.3b" model. Note that this code will first download the model which is 2.63GB in size (pgs.38-41)|
| [example_2.py](example_2.py) | Shows the structure of the "facebook/bart-large" model. Note that this code will first download the model which is 1GB in size (pg.42)|
| [example_3.py](example_3.py) | Summarizes a block of text using the "facebook/bart-large-cnn" model. Note that this code will first download the model which is 1.6GB in size (pg.44)|
| [example_4.py](example_4.py) | Shows the structure of the "bert-base-uncased" model. Note that this code will first download the model which is 440MB in size (pg.46)|
| [example_5.py](example_5.py) | Analyzes the sentiment of a line of text using the "nlptown/bert-base-multilingual-uncased-sentiment" model. Note that this code will first download the model which is 669MB in size (pg.47)|
| [example_6.py](example_6.py) | Shows the structure of the "gpt2" model and demonstrates the models ability to generate text given the beginning words of a sentence. Note that this code will first download the model which is 548MB in size (pgs.49-50)|
| [example_7.py](example_7.py) | Demonstrates an implementation of a masked self-attention mechanism (pgs.51-52)|
| [example_8.py](example_8.py) | Demonstrates use of Cohere's model and API. This requires signing up for an account with Cohere, but unlike with OpenAI there's a free API key provided, although it's rate limited to 20 calls/min for the chat functionality (pg.60)|
| [example_9.py](example_9.py) | Another demonstration of English to French translation using 3 different models: "meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", and "databricks/dolly-v2-3b". Note that this code will download each model locally and that they are 13.5GB, 14.5GB, and 5.7GB in size, respectfully. Additionally, you will need to request access to the LLaMA2 models via Hugging Face in order to use that model in this example. Once you're granted access you'll need to create a Hugging Face user access key and set it locally. Lastly, these models are large and take a long time to run if only using CPUs. This example has been modified to use my single GPU to improve speed (pgs.63-64)|
