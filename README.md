# actadd-serverless

Lots of people do [Activation Engineering](https://arxiv.org/abs/2308.10248) related stuff now. I recently helped with some ICLR rebuttal experiments that would've been substantially helped by a scalable serverless API for ActAdd. This repo allows you to set one up.


## Notes

Looks like https://cog.run/ is good for containerizing. Then we want to deploy on some serverless GPU platform that supports docker containers. Replicate supports cog natively but anything based around docker should work (runpod, vastai, etc.)

Idk where best prices will be so I'll start with replicate as it's the simplest (and probably good priced) option.


## TODO

- [ ] Deploy working cog model to replicate
- [ ] Deploy other models (llama, etc.)
- [ ] Get perf close to the wonderful 0.25$ / 1M tokens for https://replicate.com/meta/meta-llama-3-8b (maybe download and modify their cog model?)
- [ ] Re-run [grid search](https://github.com/UlisseMini/actadd-rebuttals/blob/master/figures/positive_n1024_fp_perplexity_heatmap.png) results with fast serverless API
- [ ] Ask ppl working on actadd stuff what features they'd like in serverless APIs (e.g. returning logits, perplexity, embeddings, more batching, etc.)

