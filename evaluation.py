# have this at the top to supress warnings from the imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import textstat
#import torch
from evaluate import load
#from summac.model_summac import SummaCZS
import pandas as pd
import argparse



def evaluate_generated_texts(generated_path, reference_path, output_csv=None, rouge=None, bleu=None, bertscore=None, meteor=None, summaC=None):
    # Read text files
    with open(generated_path, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f]
    with open(reference_path, "r", encoding="utf-8") as f:
        refs = [line.strip() for line in f]

    assert len(preds) == len(refs), "Mismatched number of lines in generated and reference files"

    results = []

    for i, (pred, ref) in enumerate(zip(preds, refs)):
        try:
            result = {
                "ID": i,
                "ROUGE-1": rouge.compute(predictions=[pred], references=[ref])["rouge1"],
                "ROUGE-2": rouge.compute(predictions=[pred], references=[ref])["rouge2"],
                "ROUGE-L": rouge.compute(predictions=[pred], references=[ref])["rougeL"],
                "BLEU": bleu.compute(predictions=[pred], references=[ref])["score"],
                "BERTScore_P": bertscore.compute(predictions=[pred], references=[ref], lang="en")["precision"][0],
                "BERTScore_R": bertscore.compute(predictions=[pred], references=[ref], lang="en")["recall"][0],
                "BERTScore_F1": bertscore.compute(predictions=[pred], references=[ref], lang="en")["f1"][0],
                "METEOR": meteor.compute(predictions=[pred], references=[ref])["meteor"],
                "FKGL": textstat.flesch_kincaid_grade(pred),
                "DCRS": textstat.dale_chall_readability_score(pred),
                "CLI": textstat.coleman_liau_index(pred) #,
                #"LENS": textstat.linsear_write_formula(pred) 
                #"SummaC": summaC.score([ref], [pred])[0]
            }
            results.append(result)
        except Exception as e:
            print(f"[Error on line {i}] {e}")
            continue
    
    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    return df



def main():
    parser = argparse.ArgumentParser(description="Evaluate generated texts against reference texts")
    parser.add_argument("generated_path", help="Path generated text file")
    parser.add_argument("reference_path", help="Path reference text file")
    parser.add_argument("--output_csv", help="Path to output CSV file", default=None)
    args = parser.parse_args()

    # Load metrics
    rouge = load("rouge")
    bleu = load("sacrebleu")
    bertscore = load("bertscore")
    meteor = load("meteor")

    # Load SummaC for factuality
    # summaC = SummaCZS(granularity="sentence", model_name="mnli", device="cuda" if torch.cuda.is_available() else "cpu")
    # summaC.load()

    df = evaluate_generated_texts(
        args.generated_path,
        args.reference_path,
        args.output_csv,
        rouge,
        bleu,
        bertscore,
        meteor #,
        #summaC
    )

    # Print the DataFrame
    print(df)

if __name__ == "__main__":
    main()


    """
    python evaluation.py data/test_generated_text.txt data/test_reference_text.txt --output_csv results.csv

    """
