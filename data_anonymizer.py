from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from presidio_analyzer import Pattern,PatternRecognizer

import gc
import os
import wget

import pandas

import fasttext
from google_trans_new import google_translator

spacy_supported_web_langs = ['en','zh']
spacy_supported_news_langs = ['ca', 'nl', 'el', 'it', 'hr', 'fi', 'es', 'sv', 'uk', 'de', 'ja', 'ko', 'da', 'fr', 'lt', 'mk', 'nb', 'pl', 'pt', 'ru', 'ro']

local_lang_pred_path = "./lid.176.bin"

class LanguageIdentification:

    def __init__(self):
        global local_lang_pred_path
        pretrained_lang_model = local_lang_pred_path
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1)
        return predictions

def create_models_for_config(news_langs,web_langs):
    models =[]
    model_config = {}  

    for lang in news_langs:
        model_config["lang_code"] = lang
        model_config["model_name"] = lang + "_core_news_lg"
        models.append(model_config)

    for lang in web_langs:
        model_config["lang_code"] = lang
        model_config["model_name"] = lang + "_core_web_lg"
        models.append(model_config)

    return models

def create_number_recognizer():

    number_regex = "^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$"
    numbers_pattern = Pattern(name="numbers_pattern",regex=number_regex, score = 0.2)

    number_recognizer = PatternRecognizer(supported_entity="PHONE_NUMBER", patterns = [numbers_pattern])

    return number_recognizer

number_recognizer = create_number_recognizer()

def create_analyser_with_config(news_langs,web_langs):
    global number_recognizer

    models = create_models_for_config(news_langs = news_langs,web_langs = web_langs)

    configuration = {
        "nlp_engine_name": "spacy",
        "models": models,
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    spacy_supported_langs = news_langs + web_langs

    analyzer = AnalyzerEngine(
        nlp_engine = nlp_engine, 
        supported_languages =  spacy_supported_langs,
    )

    analyzer.registry.add_recognizer(number_recognizer)

    return analyzer

def anonymize_pii_data(csv_data):

    anonymizer = AnonymizerEngine()

    lang_predictor = LanguageIdentification()

    translator = google_translator()

    fake_operators_generic = {
                        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),

                        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),

                        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),

                        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),

                        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),

                        "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "<MEDICAL_LICENSE>"}),

                        "DEFAULT": OperatorConfig("custom", {"lambda": lambda x : x})
                        }

    prev_lang = 'en'
    web_langs = ['en']
    news_langs = []
    analyzer = create_analyser_with_config(news_langs=news_langs,web_langs = web_langs)

    lang = ''
    text_to_analyze = ''
    response = ''
    analyzer_results = ''
    anonymized_results = ''
    chat_data = ['']*len(csv_data['text'])

    for index,chat_snippet in enumerate(csv_data['text']):

        text_to_analyze = chat_snippet

        (lang,_) = lang_predictor.predict_lang(chat_snippet)
        lang = lang[0][9:]  # removing __label__ frm str

        if prev_lang != lang:
            prev_lang = lang

            if lang in spacy_supported_news_langs:
                news_langs = [lang]
                web_langs = []
            elif lang in spacy_supported_web_langs:
                web_langs = [lang]
                news_langs = []
            else:
                # google transalte api is to be added
                # response = translator.translate(chat_snippet)
                chat_data[index] = chat_snippet
                continue
                text_to_analyze = response.text
                web_langs = ['en']
                news_langs = []
                lang = 'en'

            del(analyzer)
            gc.collect()

            analyzer = create_analyser_with_config(news_langs = news_langs,web_langs = web_langs)

        analyzer_results = analyzer.analyze(text_to_analyze,language=lang)

        anonymized_results = anonymizer.anonymize(
                text=text_to_analyze,
                analyzer_results=analyzer_results,            
                operators=fake_operators_generic        
            )

        chat_data[index] = anonymized_results.text

    csv_data['text'] = chat_data

    csv_data.to_csv("anonymised-chat.csv")
        
if __name__ == "__main__":

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

    if not os.path.exists(local_lang_pred_path):
        wget.download(url,local_lang_pred_path)

    csv_file_path = input("Enter the path of csv file to anonymize: ")

    csv_data = pandas.read_csv(csv_file_path)

    anonymize_pii_data(csv_data)

    