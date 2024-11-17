# Diplomatic Text Classification and Response Generation

This project is a suite of scripts that can classify diplomatic statements into one of 62 predefined categories and generate diplomatic responses or recommendations based on the classified statement.

## Overview

The project consists of three main components:

1. **Classifier**: The `classifier.py` script takes a list of diplomatic statements as input and classifies each one into one of the 62 predefined categories. It uses a pre-trained DistilBERT model for the classification task.

2. **Response Generator**: The `statement_generator.py` script takes the classified statement as input and generates either a direct diplomatic response or a set of recommendations, depending on the specified mode (`--mode res` or `--mode rec`). The generated content is designed to adhere to diplomatic principles and conventions.

3. **Trainer**: The `trainer.py` script is used to fine-tune the DistilBERT model on a dataset of diplomatic statements. It handles the data preprocessing, model training, and evaluation.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/diplomatic-text-classification.git
   cd diplomatic-text-classification
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

### Usage

#### Running the Classifier and Response Generator

1. Prepare a JSON file with a list of diplomatic statements. An example file (`examples/long_statements.json`) is provided.

2. Run the classifier and response generator:

   ```
   cat examples/long_statements.json | python classifier.py | python statement_generator.py --mode res
   ```

   This will pipe the input statements through the classifier and then the response generator in "response" mode.

   To generate recommendations instead of direct responses, use the "rec" mode:

   ```
   cat examples/long_statements.json | python classifier.py | python statement_generator.py --mode rec
   ```

#### Training the Model

1. Place your training data (a CSV file) in the `input` folder.

2. Run the training script:

   ```
   python trainer.py
   ```

   This will fine-tune the DistilBERT model on the provided dataset and save the best model checkpoint to the `output/diplomatic_text_classifier_model` directory.

#### Docker Deployment

1. Build the Docker image:

   ```
   docker build -t diplomatic-text-classification .
   ```

2. Run the containers:

   ```
   docker-compose up --build -d
   ```

   This will start the containers for the classifier, response generator, and trainer.

3. Interact with the containerized components:

   - Classify and generate responses:
     ```
     cat examples/long_statements.json | docker-compose exec -T classifier python classifier.py | docker-compose exec -T statement_generator python statement_generator.py --mode res
     ```
   - Train the model:
     ```
     docker-compose exec trainer python trainer.py
     ```

## Customization

- **Model and Tokenizer**: The project uses the DistilBERT model and tokenizer, but you can substitute these with your own preferred models and tokenizers by modifying the relevant code in `trainer.py` and `classifier.py`.
- **Label Mapping**: The label-to-ID and ID-to-label mappings are defined in the `id_to_label` and `label_to_id` dictionaries. You can customize these to fit your specific needs.
- **Input Data**: The training data is expected to be in a CSV format with "text" and "label" columns. You can modify the `load_data` function in `trainer.py` to accommodate different data formats.
- **Output Formats**: The response generation in `statement_generator.py` can be customized to produce different output formats or styles.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).