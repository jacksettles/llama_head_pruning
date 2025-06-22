# This repo goes through the basics of how to:

## 1. Perform relevance propagation
Here we use the repo associated with AttnLRP to attribute relevance to latent parts of Llama LLM models, namely the attention heads.

## 2. Prune those attention heads structurally
No prune_heads() function was ever actually implemented for these Llama models. I now know why

What we found after pruning the attention heads that were deemed "irrelevant" was that:

1. Performance on BLiMP (for testing a model's linguistic knowledge) goes down by about 10-20%
2. Model size is only reduced by 1-3%. Attention heads are typically a small part of the total 
parameters of transformer based LLMs. This is especially the case with Llama 2+ models, which 
all use Grouped Query Attention (GQA).
3. Latency actually increases, meaning the model gets slightly slower and has a lower throughput.
This is probably because the k and v tensors have to be repeated when using GQA in order to do
attention computation in parallel. Prior to pruning, these tensors are repreated at a constant amount:
every group has the same number of heads in it, so k and v need to be repeated by a factor of however many
heads there are per group. After pruning though, each group in each layer has a varying number of heads left
in it, so the model has to iterate over a list of repeat factors for each group in each layer.

## Credit where credit is due!

This repo utilizes the framework in https://github.com/rachtibat/LRP-eXplains-Transformers to perform relevance propagation on transformer based LLMs.

# Disclaimer

This code was implemented with a lot of absolute paths, so if anyone wishes to use it, just be sure to put in your own path names to your models and data. Bad practice, I know.
