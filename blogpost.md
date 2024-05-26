# Monotonic Early Exiting for Fast Inference in Transformer-based Generation 

Filipe Laitenberger, Max Belitsky, Oliver Savolainen, Mark Bodracska, Denys Sheremet<br>\{filipe.laitenberger, max.belitsky, oliver.savolainen, mark.bodracska,<br>denys.sheremet\}@student.uva.nl<br>Faculty of Artificial Intelligence<br>University of Amsterdam

$2024-05-26$


#### Abstract

Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Etiam lobortis facilisis sem. Nullam nec mi et neque pharetra sollicitudin. Praesent imperdiet mi nec ante. Donec ullamcorper, felis non sodales commodo, lectus velit ultrices augue, a dignissim nibh lectus placerat pede. Vivamus nunc nunc, molestie ut, ultricies vel, semper in, velit. Ut porttitor. Praesent in sapien. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Duis fringilla tristique neque. Sed interdum libero ut metus. Pellentesque placerat. Nam rutrum augue a leo. Morbi sed elit sit amet ante lobortis sollicitudin. Praesent blandit blandit mauris. Praesent lectus tellus, aliquet aliquam, luctus a, egestas a, turpis. Mauris lacinia lorem sit amet ipsum. Nunc quis urna dictum turpis accumsan semper.


## 1 Introduction

Large Language Models (LLMs) are showing abilities that are beyond what was deemed possible even two or three years ago [39, 1, 3, 37, These unprecedented results originate from ever-increasing model and dataset sizes, leading to immense consumption of energy and resources, as well environmental pollution [35, 21, 27]. For reference, GPT-3 consumed 1,287 MWh of energy and emitted 552 tonnes of $\mathrm{CO}_{2}$ equivalents [22], as much as 120 average cars emit per year [14].

Because of this problem, research has been invested in making LLMs more efficient. One possible way of achieving this is adaptive computation allocation, which can be realized as a network solely utilizing certain sub-networks for specialized tasks [20, skipping layers 31 or early exiting a network [26, 36, 40, 23, 12, $32,4,15,9,13$.

The current work focuses on early exiting in Transformers. We uncover that early exiting builds on the implicit assumption that the confidence of a model will increase, the more computation it performs on a token, coined the monotonicity assumption. We (1) investigate the monotonicity assumption in prominent early exiting architectures [32, 4. We conclude that a weighted crossentropy learning objective drives the model to decide on a prediction as early as possible, leading to mostly monotonic behavior after a certain layer. Furthermore, we (2) explore the hidden states of a network produced by processing sequences of different difficulty levels and examine the effects of the difficulty levels on hidden state saturation and monotonicity.

Based on the findings, we (3) propose a new early exiting mechanism that exploits monotonic behavior, called Monotonic Early Exiting (MEE). Our method exits based on the hidden states of the last $n$ layers while employing a minimum exit layer.

## 2 Related Work

### 2.1 Early Exiting in Neural Networks

While neural networks are traditionally composed of many layers that sequentially process input tokens, early exiting assumes that not all inputs need the same amount of computation. Consequently, "easy" token sequences could be output at earlier layers than "difficult" ones which need to traverse the entire network. Having been pioneered in CNN architectures [26, 36, early exiting has been studied in Transformers as well, including encoder [40, 23), encoder-decoder [12, 32, 4 ] and decoder models $[15,9,13]$.

We specifically look at two works that aim to model the confidence or uncertainty of a model when generating tokens:

CALM 32 fine-tune an LLM with a weighted cross-entropy objective that optimizes each layer to output the correct output probabilities given a shared LM-head:

$$
    \mathcal{L} = \sum^L_{i=1} \alpha_i \mathcal{L}_i
    \ \ \ 
    \text{where }
    \ \ \ 
    \alpha_i = i \bigg({\sum _{j=1}^{L} j }\bigg)^{-1}
$$

where $\mathcal{L}_{i}$ is the cross entropy loss using each layer's hidden state, and $\alpha_{i}$ favors higher layers according to the equation above.

The authors further experiment with three different confidence measures: (1) computing the word probabilities from the current hidden state after each Transformer layer and exiting if the difference between the top two probabilities exceeds a calibrated threshold; (2) computing the closing similarity between the current and last hidden state, and exiting if the similarity surpasses a calibrated threshold; (3) using a classifier that predicts the exit probability based on the current hidden state. However, when computing attention between already generated tokens that were exited earlier than the current one, CALM needs to copy hidden states individually.

FREE 4 extend CALM, trading compute adaptability for decreased overhead. Specifically, the authors reduce the number of exit points to two compared to every layer so that the model can either exit at, e.g., the fourth layer or use the entire network. Accordingly, FREE can copy missing hidden states in parallel to reduce overhead. Lastly, FREE replaces the expensively calibrated confidence thresholds used in CALM by learned ones. In addition to the weighted cross-entropy objective, FREE uses a layerwise knowledge distillation loss

$$
\mathcal{L}_{\mathrm{KD}}= \frac{1}{\left|L_S\right|} \sum _{i=1}^{L_S} \text{MSE}\left(\mathbf{H}_S^i, \mathbf{H}_D^{m(i)}\right)
$$

where $\mathbf{H}_S^i$ refers to a layer in the shallow module, i.e., the layers before the early exit point, and $\mathbf{H}_D^{m(i)}$ refers to a layer in the deep module, i.e., the layers after the early exit point. $m(i)$ either (1) maps the last layer, (2) is a uniform mapping from shallow to deep layers, or (3) maps to the closest hidden state in the deep module, i.e., $m(i)=\underset{j}{\arg \min } \text{MSE}\left(\mathbf{H}_S^i, \mathbf{H}_D^j\right)$.

## 3 Do Early-Exiting Networks Behave Monotonically?

All the methods discussed above assume that confidence evolves monotonically, i.e., that the model will be more certain of a prediction the more computation it performs on a token. This assumption is central to the functioning of early exit methods regarding the decision when to exit and whether it is sensible to exit early in the first place - it could be that the evolvement of hidden states is utterly unpredictable and does not resemble any meaningful connection to the eventual word probabilities at the final layer, i.e., the network might be a black box whose intermediate representations are meaningful to itself but not to the outside world. On the other hand, it might be that intermediate hidden states can be seen as "contemplation" of the model, or that the model even tries to decide on a prediction as early as possible in its contemplation process.

Experiment To test whether this monotonicity assumption holds, we conduct an experiment on three different settings of a T5 model - a default variant without early exiting, a CALM model optimized on the weighted cross-entropy objective, and a FREE model that uses the additional

Figure 1: The fraction of top-1 predictions that do not change after each layer, measured across the BigPatent evaluation dataset.

Figure 2: The mean and standard deviation of the end prediction across the BigPatent evaluation dataset, with the default T5 on the left, CALM in the middle, and FREE on the right.

layerwise knowledge distillation term. For each model, we use publicly available pre-trained checkpoints and evaluate each model on summarization using the BigPatent dataset 34. Specifically, we leverage the models' LM-heads after each of the twelve layers to compute two things: (1) the fraction of tokens for which the prediction does not change after the respective layer, i.e., whether the model could have exited early at the respective layer without a loss in accuracy. (2) A plot showing the mean and standard deviation of the confidence in the eventual prediction at each layer.

(3) Plots showing the top-3 predictions at each layer for individual token generations.

Results (1) Figure 1 shows that at the second layer already, the weighted cross entropy objective optimizes the model to be confident of a token, i.e. not change its top-1 prediction after layer two, in $75 \%$ of the cases. Additionally, after the fourth layer, the model keeps its prediction in $90 \%$ of the cases. Contrastingly, the default T5 and even FREE exhibit much less certainty in their predictions. Even though FREE uses the weighted cross-entropy objective as well, its additional layerwise distillation objective seems to be inhibiting the same monotonic behavior as in CALM.

(2) In addition, Figure 2 demonstrates that CALM increases monotonically in its top-1 prediction at early layers on average. Meanwhile, the vanilla T5 gains confidence much later. On the other hand, FREE displays locally monotonic behavior, i.e., its confidence increases until its first exit point, then drops and increases again until the end of the network.

(3) To illustrate this behavior in individual cases, figure 3 depicts three example forward passes of the same sequence for the three models, with the default T5 on the left, CALM in the middle, and FREE on the right. The plots exemplify the rather unpredictable behavior of the default and FREE models, while CALM decides on a prediction as early as possible. While CALM's confidence increase shows slightly non-monotonic evolvement in earlier layers, the plots exemplify its monotonicity in later layers.

Conclusion We show that the weighted cross-entropy objective encourages the model to decide on a prediction as early as possible while exhibiting monotonically increasing behavior in its predictions. These results indicate that an early exit mechanism could benefit from taking this behavior into account, exiting early based on whether it observes monotonically increasing predictions.

## 4 Sequence types: which sequences are "easy" and which are "difficult"?

In the previous section, we have seen that a Transformer model trained with a weighted crossentropy objective exhibits a monotonic pattern in token predictions. However, even having this property does not make the model confident to exit early on every possible sequence. Naturally, some sequences in a language are more ambiguous than others. For instance, the sequence One of the biggest cities in the world is New _ can be considered "easy" as the next word is most likely to be York due to this being factual knowledge. On the other hand, the sequence The students went
to _ is not that easy to predict even for a human being, as it bears an inherent degree of uncertainty without having any context.

In this section we look at the properties of the hidden states of such a monotonic network and what they can tell us about the difficulty of the input sequences.

Experiment In order to identify which sequences are "difficult" or "easy" for a model we conduct the following experiment: we select 2500 examples from the validation set of the CNN Daily Mail summarization dataset [33], iteratively feed the sequences to the model in an autoregressive manner and record the hidden states of each sequence. This procedure produces $n$ sequences. Since we use T5 large, we get 24 hidden states for each sequence. We then compute the cosine similarity between the last hidden state (which is used for the next token prediction) and hidden states after each other layer. The resulting vector shows how saturated the hidden states of an input sequence are. If the hidden states saturate quickly (become similar to the last hidden state after 4-6 layers), the benefits of further computations can be considered small. That means that the network can exit earlier on such sequences. We deem the sequences on which the network exhibits the described behavior "easy" sequences. On the other hand, the sequences that require almost a full pass through the network for the hidden states to saturate are deemed "difficult".

We compute the mean of the hidden state similarities for each sequence of tokens, sort the means in descending order, and select the first 1000 items to obtain the "easy" sequences. The same procedure but in ascending order is repeated to obtain the "difficult" sequences. Additionally, to investigate the properties of these sequences, we compute the following metrics: the index of the first layer with a saturated hidden state, the number of similar hidden states, the sequence length, and a boolean variable that indicates whether the hidden state similarities are strictly increasing after layers 0,4 and 8 . The last metrics were added to further investigate the monotonicity property and aid in coming up with better confidence measures. The similarity threshold was set at 0.9 .

Results Table 4 shows the results of the above experiment. The first finding is that (1) the mean layer index of the first similar state for "easy" sequences is 8.65 , whereas for the "difficult" sequences that number is 18.46 , which is close to the total number of layers in the model (24). In addition to that, the average number of similar hidden states in "easy" sequences is much larger than in "difficult" sequences with 14.35 and 4.54 hidden states respectively. This suggests that the hidden states of the "easy" sequences saturate much faster and do not require the full pass through the model, whereas the "difficult" sequences tend to saturate closer to the last layer.

Another significant observation is that (2) "easy" sequences tend to be considerably longer than "difficult" sequences, with 58.16 and 11.48 tokens on average respectively. This phenomenon is logical, as the space of potential tokens that can be generated is substantially larger at the beginning of the sequence generation process. In contrast, with longer sequences, the model is able to leverage contextual information, thereby making the distribution over vocabulary sharper, narrowing the scope of possible tokens.

Finally, the results on the monotonicity of the hidden state similarities indicate that $24 \%$ of "easy" sequences are strictly increasing from the layer, which is not the case for "difficult" sequences. However, the "difficult" sequences do exhibit monotonic behavior later in the network: $97 \%$ are monotonic after 4 layers and $98 \%$ are monotonic after 8 layers.

In addition to the quantitative analysis, we have plotted the hidden state similarities to inspect the differences visually. Appendix A shows examples of how the hidden states evolve over layers on some examples of sequences of both types.

|  | Easy sequences | Difficult sequences |
| :--- | :---: | :---: |
| First similar state | $8.65(1.34)$ | $18.46(0.76)$ |
| Similar states | $14.35(1.34)$ | $4.54(0.76)$ |
| Sequence length | $58.16(31.81)$ | $11.48(20.69)$ |
| Monotonic (all layers) | $24 \%$ | $3 \%$ |
| Monotonic (after layer 4) | $78 \%$ | $97 \%$ |
| Monotonic (after layer 8) | $85 \%$ | $98 \%$ |

Table 1: the properties of sequence types.

Conclusion These results indicate that an early exit mechanism could benefit from taking the sequence length into account. Additionally, it can be noted that even for the "easiest" sequences the first few layers of the network cannot be used for early exiting, which should be accounted for by the early exiting mechanism as well to not waste computation on confidence measures for these layers.

## 5 Proposed Method: Monotonic Early Exiting (MEE)

Based on our observations above, we hypothesize that an exit mechanism could improve if conditioned on multiple previous layers' hidden states. If the model is trained using the weighted cross-entropy objective, the monotonic patterns in the hidden states could inform the exit mechanism to be more confident of a decision based on the evolvement of hidden states rather than just the current hidden state. Henceforth, we develop, train, and test three new confidence measures:

LSTM-based classifier. A two-layered LSTM network [19 with the input dimensionality equal to the transformer's hidden dimensionality, and two outputs representing exit and no-exit.

Classifier based on three previous hidden states. A two-layered MLP with three times the transformer's hidden dimensionality as inputs, ReLU activation [2], the transformer's hidden dimensionality as hidden neurons, and two outputs representing exit and no-exit.

A heuristic method based on the last three top-1 softmax scores. The network makes an exit decision if and only if the last three layer's softmax scores are monotonically increasing and the current top- 1 confidence is above 0.9 . This handcrafted rule is based on the observations from our monotonicity experiments where three consecutively increasing top-1 scores above the named threshold would almost never change later on.

For all confidence measures, we set a minimum exit layer of four. We further hypothesize that the accuracy of the generation will be higher due to the increased accuracy of exit decisions and that latency will be higher due to increased computational complexity.

Comparison to FREE. FREE is less adaptive than CALM but also less computationally expensive, due to the limited number of two exit points. With the minimum exit layer of four, we aim to strike a balance between computational adaptability and complexity, informed by the behavior of monotonicity observed in the models.

Comparison to CALM. With its hidden state saturation confidence measure, CALM experiments with one measure informed by the current and previous hidden state, similar to our method. Our endeavors into hidden state saturation reveal its frequent presence. However, there may be cases without hidden state saturation even though the model is confident of a token, as very different hidden states can lead to the same softmax scores. This further motivates our classifier-based confidence measures conditioned on previous hidden states, which can recover the hidden state saturation behavior, and go beyond it if necessary. Compared to the original classifier used in the CALM paper, which didn't show great performance, our MLP takes in more hidden states as input which could give it useful information about how hidden states are changing and therefore perform better.

## 6 Experiments

Model. For all experiments, we use T5 models pre-trained on a weighted-cross entropy objective for monotonic behavior.

Datasets. We evaluate three datasets: (1) Open-book SQuAD 1.1 [30, a QA dataset out of Wikipedia articles complemented with questions and a target answer which is taken from the context article. (2) CNN/DM [18, composed of news articles and target summarizations. (3) WMT15 EN-FR 6], containing English sentences and target French translations.

Baselines. We test the three confidence measures proposed by CALM: (1) The difference between top-1 and top-2 softmax score, (2) hidden-state saturation, and (3) a classifier trained on the current hidden state. Despite describing a framework for finding threshold values, the original CALM paper doesn't have details about final threshold values, so we opted for 0.9 values which were present in the FREE repository, and also used decaying threshold temperature values of 4 for each dataset.

Novel Confidence Measures. We additionally test our three proposed confidence measures, comparing them to the three baselines. We use the loss function proposed for the CALM classifier to train our classifiers for 5 epochs.

## 7 Results

## 8 Discussion

## References

[1] Josh Achiam et al. GPT-4 Technical Report. 2024. arXiv: 2303.08774 [cs.CL].

[2] Abien Fred Agarap. "Deep Learning using Rectified Linear Units (ReLU)". In: CoRR abs/1803.08375 (2018). arXiv: 1803.08375, uRL: http://arxiv.org/abs/1803.08375.

[3] Rohan Anil et al. Gemini: A Family of Highly Capable Multimodal Models. 2024. arXiv: $2312.11805[$ cs.CL],

[4] Sangmin Bae et al. "Fast and robust early-exiting framework for autoregressive language models with synchronized parallel decoding". In: arXiv preprint arXiv:2310.05424 (2023).

[5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate". In: arXiv preprint arXiv:1409.0473 (2014).

[6] Ondřej Bojar et al. "Findings of the 2014 workshop on statistical machine translation". In: Proceedings of the ninth workshop on statistical machine translation. 2014, pp. 12-58.

[7] Tim Brooks et al. Video generation models as world simulators. https://openai . com/ index/video-generation-models-as-world-simulators. [Accessed 06-05-2024]. 2024.

[8] Tom Brown et al. "Language models are few-shot learners". In: Advances in neural information processing systems 33 (2020), pp. 1877-1901.

[9] Luciano Del Corro et al. "Skipdecode: Autoregressive skip decoding with batching and caching for efficient llm inference". In: arXiv preprint arXiv:2307.02628 (2023).

[10] Jacob Devlin et al. "Bert: Pre-training of deep bidirectional transformers for language understanding". In: arXiv preprint arXiv:1810.04805 (2018).

[11] Alexey Dosovitskiy et al. "An image is worth 16x16 words: Transformers for image recognition at scale". In: arXiv preprint arXiv:2010.11929 (2020).

[12] Maha Elbayad et al. "Depth-adaptive transformer". In: arXiv preprint arXiv:1910.10073 $(2019)$.

[13] Mostafa Elhoushi et al. "Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding". In: arXiv preprint arXiv:2404.16710 (2024).

[14] United States Environmental Protection Agency EPA. Greenhouse Gas Emissions from a Typical Passenger Vehicle - US EPA - epa.gov. [Accessed 13-05-2024]. 2023. URL: https: //www . epa . gov/greenvehicles/greenhouse-gas - emissions - typical - passenger vehicle\#typical-passenger.

[15] Mor Geva et al. "Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space". In: arXiv preprint arXiv:2203.14680 (2022).

[16] Rohit Girdhar et al. "Imagebind: One embedding space to bind them all". In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, pp. 1518015190.

[17] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning.http://www.deeplearningbook. org. MIT Press, 2016.

[18] Karl Moritz Hermann et al. "Teaching machines to read and comprehend". In: Advances in neural information processing systems 28 (2015).

[19] Sepp Hochreiter and Jürgen Schmidhuber. "Long Short-term Memory". In: Neural computation 9 (Dec. 1997), pp. 1735-80. DOI: $10.1162 /$ neco.1997.9.8.1735

[20] Albert Q. Jiang et al. Mixtral of Experts. 2024. arXiv: 2401.04088 [cs.LG]

[21] Pengfei Li et al. "Making ai less" thirsty": Uncovering and addressing the secret water footprint of ai models". In: arXiv preprint arXiv:2304.03271 (2023).

[22] Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. "Estimating the carbon footprint of bloom, a 176b parameter language model". In: Journal of Machine Learning Research 24.253 (2023), pp. 1-15.

[23] Sourab Mangrulkar, Ankith MS, and Vivek Sembium. "BE3R: BERT based Early-Exit Using Expert Routing". In: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022, pp. 3504-3512.

[24] Brandon McKinzie et al. MM1: Methods, Analysis Insights from Multimodal LLM Pretraining. 2024. arXiv: 2403.09611 [cs.CV].

[25] Sachin Mehta et al. OpenELM: An Efficient Language Model Family with Open Training and Inference Framework. 2024. URL: https://arxiv.org/abs/2404.14619.

[26] Priyadarshini Panda, Abhronil Sengupta, and Kaushik Roy. "Conditional deep learning for energy-efficient and enhanced pattern recognition". In: 2016 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE. 2016, pp. 475-480.

[27] David Patterson et al. "Carbon emissions and large neural network training". In: arXiv preprint arXiv:2104.10350 (2021).

[28] Alec Radford et al. "Improving language understanding by generative pre-training". In: $(2018)$.

[29] Colin Raffel et al. "Exploring the limits of transfer learning with a unified text-to-text transformer". In: Journal of machine learning research 21.140 (2020), pp. 1-67.

[30] Pranav Rajpurkar et al. "Squad: 100,000+ questions for machine comprehension of text". In: arXiv preprint arXiv:1606.05250 (2016).

[31] David Raposo et al. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. 2024. arXiv: 2404.02258 [cs.LG],

[32] Tal Schuster et al. "Confident adaptive language modeling". In: Advances in Neural Information Processing Systems 35 (2022), pp. 17456-17472.

[33] Abigail See, Peter J. Liu, and Christopher D. Manning. "Get To The Point: Summarization with Pointer-Generator Networks". In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vancouver, Canada: Association for Computational Linguistics, July 2017, pp. 1073-1083. DOI: 10.18653/v1/P17-1099. URL: https://www.aclweb.org/anthology/P17-1099.

[34] Eva Sharma, Chen Li, and Lu Wang. "BIGPATENT: A large-scale dataset for abstractive and coherent summarization". In: arXiv preprint arXiv:1906.03741 (2019).

[35] Emma Strubell, Ananya Ganesh, and Andrew McCallum. "Energy and policy considerations for deep learning in NLP". In: arXiv preprint arXiv:1906.02243 (2019).

[36] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. "Branchynet: Fast inference via early exiting from deep neural networks". In: 2016 23rd international conference on pattern recognition (ICPR). IEEE. 2016, pp. 2464-2469.

[37] Hugo Touvron et al. "Llama: Open and efficient foundation language models". In: arXiv preprint arXiv:2302.13971 (2023).

[38] Ashish Vaswani et al. "Attention is all you need". In: Advances in neural information processing systems $30(2017)$.

[39] Jason Wei et al. "Emergent abilities of large language models". In: arXiv preprint arXiv:2206.07682 $(2022)$.

[40] Ji Xin et al. "BERxiT: Early exiting for BERT with better fine-tuning and extension to regression". In: Proceedings of the 16th conference of the European chapter of the association for computational linguistics: Main Volume. 2021, pp. 91-104.

