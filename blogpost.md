# Monotonic Early Exiting for Fast Inference in Transformer-based Generation 

### _Filipe Laitenberger, Max Belitsky, Oliver Savolainen, Mark Bodracska, Denys Sheremet_

---
**Exploring Efficiency in Large Language Models: A Journey into Adaptive Computation and Monotonic Early Exiting**

Large Language Models (LLMs) exhibit exceptional performance. The primary factor behind this rapid advancement is the substantial increase in the size of the models and datasets used. By expanding the models and providing them with larger datasets, we have been able to achieve unprecedented levels of performance.

However, this progress comes at a significant cost. Training these massive models requires an enormous amount of energy and resources, which in turn leads to substantial environmental impact. For example, GPT-3 consumed 1,287 MWh of energy and emitted 552 tonnes of CO₂ equivalents during its training process. That's about as much carbon dioxide as 120 average cars emit in a year. Furthermore, many applications, such as autonomous driving or real-time voice assistants, cannot afford high latency when generating predictions.

How can we continue to advance AI while avoiding a high-latency bottleneck? One promising direction is to make models allocate their resources more efficiently. Imagine we could teach a model to be smart about how it uses its computational power — only activating certain parts of its network when needed, or knowing when it’s done processing a piece of information early. Drawing an analogy to the human brain, you might think of it as the model being able to choose how long it ponders about a certain decision. This concept is known as adaptive computation allocation.

One promising approach within this concept is called early exiting. Instead of running every piece of input through every layer of a model, the model can decide to "exit" early if it’s confident enough in its prediction. This way, we save computational resources by not over-processing data. 

In this work, we focus on Transformer models, the backbone of most state-of-the-art language models. For early exiting to work effectively, there’s an underlying assumption: the more a model processes a token, the more confident it becomes in its prediction. We call this the *monotonicity assumption*. Essentially, it means that as the model processes information layer by layer, confidence should steadily increase without decreasing again. [also, say that it shouldnt change its prediction]

In the first part of this blog post, we want to give an introduction to early exiting, explaining it and its evolution in more depth. 
Moreover, we performed interesting investigations into the inner workings of common early exiting architectures, presented in the second part. 
Lastly, based on our investigations, we've come up with new ways of improving early exiting architectures, which we will show at the end.

We struture the rest of this blog post into three parts: 
1. The first part explains early exiting and its evolution in more depth.
2. In the second part, we investigate deeper into early exiting and its monotonicity assumption. We specifically test for which architectures and especially loss functions it holds and for which it doesn't. We furthermore delve into how neural networks process "easy" and "hard" sentences, gaining insights into when early exiting makes more sense and when it doesn't.
3. Based on our insights from section 2, we experiment with new early exiting methods.

## 1. What is Early-Exiting in Neural Networks?

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_framework.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The overview of the early exiting framwork.</em>
   <br />
</p>

In traditional neural networks, input tokens pass sequentially through many layers, each adding more processing and refining the output. However, early exiting suggests that not all inputs require the same level of computation. Some "easy" token sequences might be confidently processed by the earlier layers, while more "difficult" sequences need to go through the entire network. This idea isn't new—it was first explored in convolutional neural networks (CNNs) and has since been applied to Transformer models as well.
Two notable studies that delve into early exiting by modeling the confidence or uncertainty of a model when generating tokens are CALM and FREE.

#### CALM: Confident Adaptive Language Modelling

The CALM method fine-tunes a large language model (LLM) using a weighted cross-entropy objective. This objective optimizes each layer to output the correct probabilities for the next token, using a shared language model head across all layers. The loss function for this method is given by:

$$
\mathcal{L} = \sum^L_{i=1} \alpha_i \mathcal{L}_i
\ \ \
\text{where }
\ \ \
\alpha_i = i \ / \ {\sum _{j=1}^{L} j }
$$

[TODO: check whether all of the formulas are correct]

Here, $\mathcal{L}_i$ represents the cross-entropy loss at layer $i$, and $\alpha_i$ are weights that favor higher layers.

CALM explores three different ways to measure confidence:

**Probability Thresholding**: After each Transformer layer, the model calculates word probabilities from the current hidden state and exits if the difference between the top two probabilities exceeds a calibrated threshold.

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_softmax.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The overview of the early exiting framwork.</em>
   <br />
</p>
   
**Hidden State Similarity**: The model computes the similarity between the current hidden state and the previous one, exiting if the similarity surpasses a calibrated threshold.

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_hidden_similarity.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The overview of the early exiting framwork.</em>
   <br />
</p>
   
**Classifier Prediction**: A classifier predicts the probability of exiting based on the current hidden state.

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_mlp.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The overview of the early exiting framwork.</em>
   <br />
</p>

A challenge with CALM is handling attention between tokens when some have exited earlier than others, requiring individual copying of hidden states.

#### FREE: Fast and Robust Early Exiting

FREE extends CALM by balancing computational adaptability with reduced overhead. Instead of providing an exit point after every layer, FREE restricts it to two specific points—early and at the end of the network. For instance, the model might exit at the fourth layer or use the entire network. This approach allows for copying of missing hidden states in parallel, reducing the computational burden.

FREE also replaces the calibrated confidence thresholds in CALM with learned ones. In addition to the weighted cross-entropy objective, FREE incorporates a layer-wise knowledge distillation loss:

$$
\mathcal{L}_{KD}= \frac{1}{L_S} \sum _{i=1}^{L_S} \text{MSE} \left( \mathbf{H}_S^i, \mathbf{H}_D^{m(i)} \right)  
$$

In this equation, $\mathbf{H}_S^i$ denotes the hidden state in the shallow module (before the early exit point), and $\mathbf{H}_D^{m(i)}$ denotes the corresponding hidden state in the deep module (after the early exit point). The mapping $m(i)$ can take various forms:

[Usually the last layer of the shallow module is mapped to the last layer of the deep module and so on.]

## 2. What happens inside an Early-Exiting network?

### Do Early-Exiting Networks Behave Monotonically?

Now, let's dive deeper into a key assumption underlying early exiting methods: the monotonicity assumption. This assumption posits that as a model processes a token through more layers, its confidence in the prediction for that token should steadily increase. In simpler terms, the more computation the model performs on a token, the more certain it becomes about a prediction.

But why is this assumption so important? Imagine if a model's confidence didn't increase with more processing. Then it would not make sense to exit the network early - the model could be confident of a token at one layer, and then change its prediction entirely in the next layer. That is to say, there would be no way of being sure that the model's prediction at a certain layer is truly reliable. On the other hand, it seems intuitive that, the more the model thinks about something, the more reliable its prediction is.

### Testing the Monotonicity Assumption

The early exiting methods implicitly assume the monotonicity property, but they don’t test whether it actually holds. So to see if this monotonicity assumption holds, we conduct experiments with three different versions of the T5 model.
We test the monotonicity assumption on the default T5 model, which does not use early exiting, the CALM model, and the FREE model. 
We use the BigPatent dataset, which is commonly used for summarization tasks. 
 
The monotonicity assumption states that when a model makes a top prediction at a given layer, this prediction will persist as top prediction through subsequent layers, with the model becoming increasingly confident in this prediction as it progresses.
In other words, there are two parts to this assumption: the prediction remains the same after a given layer, and the model (monotonically) becomes more confident in that prediction with each additional layer. We test both parts of the monotonicity assumption with the following two experiments. 


**Fraction of Stable Predictions**: We measure the fraction of tokens for which the top-1 prediction remains unchanged after each layer. If this fraction is high for a certain layer, it means that at that layer the model has often made a prediction and doesn't change it in later layers.

<p 
   align='center'
>
   <img 
      src="https://github.com/Thiggel/Monotonic-Early-Exit/blob/main/img/nochange-1.png?raw=true" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The fraction of predictions that do not change after each respective layer.</em>
   <br />
</p>

[change it so that in both plots the default network is called either "Default" or "Vanilla"]

From the results, we can see that the CALM model often makes a final prediction very early on in the network, and rarely changes its prediction. 
In contrast, the default and FREE models change their predictions much more often in later layers. This indicates that the CALM model may be a better candidate for using early exiting, because exiting early is much less likely to change the prediction for a given token. 


**Confidence Over Layers**: We also plot the mean and standard deviation of the model's confidence in its final prediction across layers. This helps us visualize how the model's confidence evolves as it processes more layers.

<p 
   align='center'
>
   <img 
      src="https://github.com/Thiggel/Monotonic-Early-Exit/blob/main/img/top1_curves-1.png?raw=true" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 2:</b> The mean and standard deviation of the confidence curves for the eventual predictions, plotted at each layer.</em>
   <br />
</p>

Figure 2 demonstrates that CALM shows a clear monotonic increase in confidence as it processes more layers. The default model, however, gains confidence much later in the network, while FREE shows a more complex pattern: its confidence increases until the first exit point, drops slightly, and then increases again towards the end. Technically speaking, this means that the monotonicity assumption holds for both the default and CALM models, but the CALM model gains confidence much earlier in the network and therefore seems to be more fit for being used with early exiting. 

The results of these experiments show that the monotonicity assumption holds best for the CALM method. 
This suggests that CALM with its cross-entropy objective may be the best candidate for early exiting. 
With this method, exiting early at a certain layer has a high likelihood of yielding the same prediction as continuing through all layers.

### Easy and Difficult Sequences: Understanding Hidden State Saturation

In the previous section, we explored how a Transformer model, trained with a weighted cross-entropy objective, exhibits a monotonic pattern in token predictions. However, this doesn’t mean the model can confidently exit early on every possible sequence. Some sequences are more straightforward ("easy") while others are more ambiguous ("difficult"). For instance, consider the sequence "One of the biggest cities in the world is New _". The next word is likely to be "York" because it's a well-known fact. In contrast, the sequence "The students went to _" is harder to predict without additional context.

### Investigating Hidden States: Easy vs. Difficult Sequences

To understand the properties of hidden states in a monotonic network and their relation to sequence difficulty, we conducted a second experiment. We used the CNN Daily Mail summarization dataset, selecting 2500 examples from the validation set. These sequences were fed into a T5 model in an autoregressive manner, recording the hidden states at each layer. This procedure generated 24 hidden states per sequence, given the T5 model has 24 layers.

#### Experiment Details

1. **Cosine Similarity of Hidden States**: We computed the cosine similarity between the last hidden state (used for next token prediction) and the hidden states from each previous layer. This similarity vector shows how quickly hidden states saturate. If hidden states become similar to the final state after just a few layers (e.g., 4-6 layers), further computation yields minimal benefits. We classify such sequences as "easy". Conversely, sequences requiring almost the entire network to saturate are labeled "difficult".

2. **Identifying Easy and Difficult Sequences**: We calculated the mean similarity for each sequence. We labeled the 1000 sequences with the highest mean similarity as "easy" and the 1000 sequences with the lowest mean similarity as "difficult".

3. **Metrics Computed**: We analyzed the following metrics:
   - **Index of the First Saturated Layer**: The layer where hidden states first reach high similarity.
   - **Number of Similar Hidden States**: The total number of layers where hidden states are similar to the final state.
   - **Sequence Length**: Average length of sequences.
   - **Monotonicity Indicator**: Whether hidden state similarities strictly increase after layers 0, 4, and 8.

#### Results and Observations

1. **Layer Index and Hidden State Saturation**:
   - Easy sequences saturate much earlier, around layer 8.65 on average.
   - Difficult sequences saturate much later, around layer 18.46, close to the model’s final layer.

2. **Number of Similar Hidden States**:
   - Easy sequences have a larger number of similar hidden states (14.35 on average).
   - Difficult sequences have fewer similar hidden states (4.54 on average).

3. **Sequence Length**:
   - Easy sequences tend to be longer (58.16 tokens on average). 
   - Difficult sequences are shorter (11.48 tokens on average).

4. **Monotonicity of Hidden States**:
   - A significant portion of easy sequences shows strictly increasing hidden state similarity from the start.
   - Difficult sequences exhibit monotonic behavior later, with a noticeable increase after layers 4 and 8.

[We should explain what these results imply (e.g. 3. Can be explained by that long sequences have more context and therefore are easier to predict. 1&2 imply that there exist sequences for which we can confidently early exit. 4 provides additional evidence for the monotonicity assumption.)]

#### Visualizing Hidden State Evolution

To better illustrate these findings, we plotted the hidden state similarities for various sequences, showing how the hidden states evolve across layers for both easy and difficult sequences. These visualizations highlight the differences in saturation and monotonic behavior.

<p 
   align='center'
>
   <img 
      src="img/sequence_types/example_93.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <img 
      src="img/sequence_types/example_58.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure N:</b> Examples of how hidden state similarities evolve over layers in "easy" (green) and "difficult" sequences (blue).</em>
   <br />
</p>

### Conclusion

Our analysis indicates that an early exit mechanism can benefit from considering sequence length and the early layers’ computation. Even the easiest sequences require a few initial layers before reaching high confidence, suggesting that the first few layers should not be used for early exits to avoid unnecessary computation on confidence measures. This understanding helps refine early exiting mechanisms, optimizing resource use without sacrificing performance.

## 3. Experimenting with New Early Exiting Methods Based on Our Insights
### Making use of the monotonicity assumption for early exiting

After our deep dive into the behavior of hidden states, we realized there’s a lot of potential in leveraging multiple layers' hidden states to improve the early exit decision process. If we train our model with a weighted cross-entropy objective, it encourages a sort of "confidence buildup" layer by layer. This observation leads us to hypothesize that using a combination of hidden states from previous layers, rather than just the current one, could make our exit mechanism more reliable.

We design, train, and test three new confidence measures to put this theory to the test:

**Three-Previous-Hidden-States Classifier**: Here, we use a two-layered MLP that takes in the concatenation of the last three layers' hidden states.

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_3_mlp.png" 
      alt="An MLP that is fed with three hidden states and produces an early-exit decision" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> An MLP that is fed with three hidden states and produces an early-exit decision.</em>
   <br />
</p>

**LSTM-based Classifier**: This method employs a two-layered LSTM network. This is similar to the Three-Previous-Hidden-States Classifier, but it utilizes the recurrent architecture to look at all of the previous hidden states. 

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_lstm.png" 
      alt="An LSTM that sequentially processes the hidden state at each layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> An LSTM that sequentially processes the hidden state at each layer.</em>
   <br />
</p>

**Heuristic Based on Top-1 Softmax Scores**: This heuristic exits if the last three layers' top-1 softmax scores are monotonically increasing and the current top-1 confidence exceeds 0.9. This heuristic is grounded in our observations from the monotonicity experiments, where such patterns almost always indicated a stable prediction.

<p 
   align='center'
>
   <img 
      src="img/diagrams/early_exit_3_softmax.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> A heuristic that exits if the last three top-1 predictions are increasing and the model is at least 90% confident.</em>
   <br />
</p>

### Comparing confidence measures
We use T5 models pre-trained with a weighted cross-entropy objective, tuned for monotonic behavior, to evaluate our methods. 
To get insight into how our proposed new early-exiting methods work in various scenarios, we want to test them on Question-Answering, Summarization, and Translation, which are the most popular tasks for LLMs.

1. **Open-book SQuAD 1.1**: A Question-Answering dataset sourced from Wikipedia articles, supplemented with questions and corresponding answers from the context.
2. **WMT15 EN-FR**: This dataset contains English sentences paired with their French translations.
3. **CNN/DM**: A summarization dataset composed of news articles and their target summaries.

#### Baselines and Novel Measures

We compare our new confidence measures against the three baseline methods from CALM:

1. **Top-1 vs. Top-2 Softmax Score Difference**: Exits when the top-1 and top-2 score difference exceeds a threshold.
2. **Hidden-State Saturation**: Measures the change in hidden state similarity across layers.
3. **Single Hidden State Classifier**: A classifier trained on the current hidden state.

To summarize, we are evaluating six different confidence measures, three of which are classifiers that need to be trained before. Therefore, we first train these for five epochs, driving them to output true whenever the current layer's prediction matches the ground-truth target, and "false" otherwise.

The results on the three datasets are shown in the plots below. More specifically, these plots show each confidence measure's performance vs. its latency, i.e., how good it is vs. how fast it is. Performance is measured in different ways (F1, BLEU, and ROUGE-1), but generally, the higher the performance, the more similar its prediction is to the ground truth. Furthermore, latency is measured as the number of tokens the model produces per second. Thus, the closer a point is to the top-right corner of the plot, the better. 

<p 
   align='center'
>
   <img 
      src="img/SQUAD F1 Score-1.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The results on SQuAD (Question-Answering), measured in F1-score vs. produced samples per second.</em>
   <br />
</p>

<p 
   align='center'
>
   <img 
      src="img/ISWLT BLEU Score-1.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The results on IWSLT (Translation), measured in BLEU-score vs. produced samples per second.</em>
   <br />
</p>

<p 
   align='center'
>
   <img 
      src="img/CNNDM ROUGE-1 Score-1.png" 
      alt="Fraction of predictions that do not change after each respective layer" 
      style="
        width: 800px; 
        max-width: 100%;
        height: auto;
      "
   />
   <br />
   <em><b>Figure 1:</b> The results on CNN/DM (Summarization), measured in ROUGE-1-score vs. produced samples per second.</em>
   <br />
</p>


These results show that our proposed confidence measures exhibit much better performance than CALM's confidence, i.e., they are much closer to the top (and the no-early-exiting baseline) than the original measures. However, they also exhibit high latency, mostly even slower than no early exiting.

Where does this leave us? We've gained the insight that making use of monotonicity does benefit the model's performance. Nevertheless, optimizing these confidence measures would still be crucial to making them useful for real-world applications. It seems to make sense to condition a classifier on multiple past hidden states since the classifier can truly make use of this information and arrive at much better-informed exit decisions. On the other hand, the downside of using multiple hidden states seems to be that there is that it creates a lot more overhead as the classifier has to process more information and more memory is used for saving past hidden states.

## Wrapping Up

In this blog post, we deep-dove into the monotonous behavior of early-exiting models, first hypothesizing and then showing how and in what situations they become increasingly more confident of a prediction over time. Based on this, we designed new confidence mechanisms that make use of this property and actually ended up displaying much higher accuracy compared to other confidence measures. Nonetheless, there are still questions remaining that should be addressed through further research:

1. Can we make use of monotonicity and still have the performance benefits of CALM's confidence methods that only look at the current hidden state? This is most likely an engineering problem which requires many additional optimizations that we did not have the time to implement yet.
2. Is early exiting the best way of making a model more efficient? We showed that early-exiting crucially depends on this assumption, and that making the model decide on a decision as early as possible benefits the exit mechanism. In other words, we are restricting the model to steer its thinking process into one direction quickly, which takes away the ability to freely ponder. Perhaps, models could benefit from being able to randomly contemplate many different things. There is two further directions we want to mention here: (1) This blogpost^[1] and this paper^[2] show that the first few layers of LLMs exhibit quite random behavior while the rest is more predictable. This leads us to hypothesize that it would be better to constrain the weighted cross entropy objective to just optimizing the last 70% of the model's layers, restricting it less and giving it more time to ponder and explore different directions at first. (2) A very different approach is Mixture-of-Depths^[3] which skips layers instead of exiting altogether. This alleviates the model of having to be monotonous and hence doesn't restrict it at all. It can exhibit completely random behavior, think in many different ways, and at the same time decide to skip certain parts, "specializing" different stages for different processing steps of a token.

## References

[1] Nostalgebraist, “interpreting GPT: the logit lens,” Aug. 31, 2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

[2] C. Wendler, V. Veselovsky, G. Monea, and R. West, “Do llamas work in English? on the latent language of multilingual transformers,” arXiv.org, Feb. 16, 2024. https://arxiv.org/abs/2402.10588

[3] D. Raposo, S. Ritter, B. Richards, T. Lillicrap, P. C. Humphreys, and A. Santoro, “Mixture-of-Depths: Dynamically allocating compute in transformer-based language models,” arXiv.org, Apr. 02, 2024. https://arxiv.org/abs/2404.02258



[1] Josh Achiam et al. GPT-4 Technical Report. 2024. arXiv: 2303.08774 [cs.CL].

[2] Abien Fred Agarap. "Deep Learning using Rectified Linear Units (ReLU)". In: CoRR abs/1803.08375 (2018). arXiv: 1803.08375, URL: http://arxiv.org/abs/1803.08375.

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

