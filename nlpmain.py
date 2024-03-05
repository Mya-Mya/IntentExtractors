from argparse import ArgumentParser
from pathlib import Path
from iie import IIE
from nlpie import NLPIE
from server import launch

if __name__ == "__main__":
    parser = ArgumentParser("Intent Extractor NLP Main Component")
    parser.add_argument(
        "--sbert_path",type=Path, required=True,help="Path to the Sentence-BERT Model Pretrained Directory")
    parser.add_argument(
        "--qamodel_path",type=Path, required=True,help="Path to the QA Model Pretrained Directory")
    args = parser.parse_args()
    
    ie = NLPIE(sbert_name=args.sbert_path, qamodel_name=args.qamodel_path)
    launch(ie)