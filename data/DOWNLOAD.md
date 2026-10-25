# Official data (never redistributed)
- WILDS Amazon v2.1: `from wilds import get_dataset; get_dataset('amazon', download=True, root_dir=...)`.
- Yahoo Answers Topics: `datasets.load_dataset('yahoo_answers_topics')`.
- CIFAR-100 anchor logits and labels: data/c100_cells.
Web logits in data/web_logits carry the official row of every item and the
SHA-256 of its label vector; code/canonical/join_web_labels.py re-joins and
verifies the labels.
