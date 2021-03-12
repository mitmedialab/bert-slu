# BERT for Intent Recognition

Code accompanying [Practical Guidelines for Intent Recognition: BERT with Minimal Training Data Evaluated in Real-World HRI Application](https://dl.acm.org/doi/10.1145/3434073.344467)


## Citation

Matthew Huggins, Sharifa Alghowinem, Sooyeon Jeong, Pedro Colon-Hernandez, Cynthia Breazeal, and Hae Won Park. 2021. Practical Guidelines for Intent Recognition: BERT with Minimal Training Data Evaluated in Real-World HRI Application. In Proceedings of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (HRI '21). Association for Computing Machinery, New York, NY, USA, 341–350. DOI:https://doi.org/10.1145/3434073.3444671

## Setup
Requires Python 3
pip install -r requirements.txt

## Getting Started

python main-eval.py --data_path "./snips/"  --epochs 1 --batch_size 32 --output_dir "./model_save_snips_1ep/"
python eval.py --data_path "./snips/" --epochs 1 --batch_size 32  --output_dir "./model_save_snips_1ep/"