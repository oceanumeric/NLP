## Train GPT-2 from scratch

Papers and citations

Bengio, Y., Ducharme, R., & Vincent, P. (2000). A neural probabilistic language model. Advances in neural information processing systems, 13.

10075 

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.

46775 

Mikolov, T., Karafiát, M., Burget, L., Cernocký, J., & Khudanpur, S. (2010, September). Recurrent neural network based language model. In Interspeech (Vol. 2, No. 3, pp. 1045-1048).

7125

Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.

4553


Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259.

6892

Ioffe, S., & Szegedy, C. (2015, June). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456). pmlr.

47399

Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.

5400

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

76820


rambling: 

--- basic model

iteration 4500, train loss 2.2300, test loss 2.2466

Wife?
Whord
Thow aff is and the aceet bobe toe.
Sthr-and thalllands:
Waith foulqoutht. War dilthoate


--- residual net + block 

iteration 4500, train loss 1.9992, test loss 2.1048

And they bridcewill, is by be madisel bube a enavegry'd the gatanse:
Warthat us him to barddels
Hay,

--- layernorm

iteration 4500, train loss 1.9911, test loss 2.0899

And they bridce.

STAULOLUS:
KING Proke? you eyanthrud my dagatands:
Warthis us combe. Warderlascane


--- full techniques (drop out + residual + layer norm + attention)

iteration 4500, train loss 0.6588, test loss 1.7911


And Lewis  here.

KING EDWARD IV:
Then but false, trumps them back to your fathers,
For Warwick, Harrarand rose again a world,
And in the srift command, but from my sheat,
I will not dream of the world's thought,
But be revenged, which, without die condemn'd.

Messenger:
My l