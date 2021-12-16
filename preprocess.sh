fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "enthymemes/train.bpe" \
  --validpref "enthymemes/val.bpe" \
  --destdir "enthymemes-bart/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
