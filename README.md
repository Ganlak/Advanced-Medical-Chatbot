# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.8 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command to download the chromadb
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

```bash
# Finally run the following command
pip -freeze > requirements.txt
```

```bash
# Execute the following command in the command line to build the Docker image
docker build -t my-flask-app .
```

```bash
# Execute the following command to launch the chatbot application
docker run -p 5000:5000 my-flask-app
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- chromadb
- Dockerfile


