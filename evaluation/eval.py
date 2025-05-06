#import argparse
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as bert_score
import textstat as ts
import pandas as pd
#import lens_metric as lens
#from alignscore import AlignScore
# from sumac.score import SummaCZS
from tqdm import tqdm

import json
import evaluate

def read_data(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = []
            for line in f:
                try:
                    data.append(json.loads(line)["summary"])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, Line: {line}")
            return data
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            # remove the start and end quotes from each line
            return [line.strip()[1:-1] for line in f.readlines() if line.strip()]
        
def cal_meteor(preds, refs):
  meteor = evaluate.load('meteor')
  scores = meteor.compute(predictions=preds, references=refs)["meteor"]
  return scores

def main(pred_file, ref_file, output_file):
    
    print("Beginning evaluation...")

    predictions = read_data(pred_file)
    references = read_data(ref_file)

    print(f"Predictions: {len(predictions)}")
    print(f"References: {len(references)}")

    # if len(predictions) != len(references):
    # set the length of the longer list to the length of the shorter one
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    print(f"Adjusted Predictions: {len(predictions)}")
    print(f"Adjusted References: {len(references)}")

    assert len(predictions) == len(references), "Mismatch in prediction and reference counts!"

    print("Rouge scoring...")
    results = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoother = SmoothingFunction().method4
    #align = AlignScore()
    # sumac = SummaCZS()

    print("Evaluating...")
    i = 1
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        r_scores = scorer.score(ref, pred)
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother)
        #meteor = single_meteor_score(ref, pred)
        meteor = cal_meteor([pred], [ref])
        P, R, F1 = bert_score([pred], [ref], lang="en", verbose=False)

        try:
            fkgl = ts.flesch_kincaid_grade(pred)
            dcrs = ts.dale_chall_readability_score(pred)
            cli = ts.coleman_liau_index(pred)
        except Exception:
            fkgl = dcrs = cli = None
        print (f"Prediction {i}")
        i += 1

        #lens_score = lens.score(pred, ref)
        #align_score = align.score([pred], [ref])["scores"][0]
        # summac_score = sumac.score([{"src": "", "cand": pred, "ref": ref}])[0]["scores"]["factuality"]

        results.append({
            "ROUGE-1": r_scores['rouge1'].fmeasure,
            "ROUGE-2": r_scores['rouge2'].fmeasure,
            "ROUGE-L": r_scores['rougeL'].fmeasure,
            "BLEU": bleu,
            "METEOR": meteor,
            "BERTScore_F1": F1[0].item(),
            "FKGL": fkgl,
            "DCRS": dcrs,
            "CLI": cli #,
            #"LENS": lens_score #,
           # "AlignScore": align_score #,
            #"SummaC": summac_score,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}\n")
    print(df.describe())

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Evaluate system summaries against gold references.")
    # parser.add_argument("--pred_file", type=str, required=True, help="File with system-generated summaries (one per line).")
    # parser.add_argument("--ref_file", type=str, required=True, help="File with gold reference summaries (one per line).")
    # parser.add_argument("--output_file", type=str, default="evaluation_results.csv", help="Where to save the evaluation scores.")
    # args = parser.parse_args()


    pred_file = "elife_summaries100.txt"
    ref_file = "data/elife_references100.json"
    output_file = "evaluation_results.csv"


    main(pred_file, ref_file, output_file)
