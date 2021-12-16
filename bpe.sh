for SPLIT in train val
do
  for LANG in source target
  do
    python3 -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "enthymemes/$SPLIT.$LANG" \
    --outputs "enthymemes/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
