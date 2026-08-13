# SigLIP Vision-Encoder Fine-Tuning

## Initialization and model setup

- The starting multimodal model is LLaVA-OneVision Qwen2 0.5B SI, loaded from `/scratch/common_models/llava-onevision-qwen2-0.5b-si-hf`.
- Fine-tuning is applied only to LLaVA's SigLIP vision tower; the remaining LLaVA parameters are frozen.
- The model is initially loaded with FP16 weights. For the configured full fine-tuning run (LoRA and QLoRA disabled), the vision tower is converted to FP32 before training.
- The execution device is selected in the order CUDA, Apple MPS, then CPU. On CUDA, Hugging Face `device_map="auto"` is used.
- The vision-tower embedding layer is frozen.
- The vision encoder contains 26 transformer layers. The first 9 layers are frozen and the final 17 layers are trainable (`trainable_vision_layers=17`). This preserves the earlier low-level visual representations while adapting the later semantic layers.
- A separate pretrained SigLIP model (`siglip-so400m-patch14-384`) supplies the text encoder and tokenizer/processor.
- The complete SigLIP text model is frozen and kept in evaluation mode throughout training. Text embeddings are computed without gradient tracking.
- The vision and text hidden dimensions are checked for compatibility during initialization.
- Image representations are obtained by mean-pooling the vision tower's final hidden states across all image tokens. The original SigLIP vision pooling head is not used.
- The SigLIP logit scale and logit bias are copied from the pretrained SigLIP model and remain trainable. The logit scale is capped at `log(100)` before exponentiation for numerical stability.
- A trainable linear concept bottleneck maps each pooled image representation to 14,094 concept logits.
- Only the concept bottleneck is enabled. The object bottleneck (3,679 classes) and predicate bottleneck (18 classes) are disabled.
- LoRA and QLoRA are disabled for the reported run; therefore, the selected vision layers are directly fine-tuned.
- Initialization sanity checks verify the frozen/trainable parameter configuration, hidden-size compatibility, and trainability of the concept head, logit scale, and logit bias.

## Data preparation

- Samples are read from `concept_metadata.json` using the predefined `train`, `valid`, and `test` splits.
- Each image is associated with all of its available captions and multi-hot target vectors for concepts, objects, and predicates.
- Images are converted to RGB, resized to `384 × 384`, converted to tensors, and normalized channel-wise using mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`.
- The batch size is 8 for training, validation, and testing.
- Training samples are shuffled; validation and test samples are not shuffled.
- Data loading uses `num_workers=0`.
- Variable numbers of captions per image are retained by the custom collation function, while image and target tensors are stacked.
- Concept class imbalance is handled with a per-concept positive weight computed from the training split as `negative_count / positive_count`.
- Concept positive weights are capped at 20. Concepts without both positive and negative training examples receive a weight of 1.

## Training process

- For every batch, all captions belonging to the batch images are flattened and encoded by the frozen SigLIP text encoder.
- An image-caption matching matrix is constructed across the complete batch: captions belonging to an image receive label `+1`, while captions belonging to other images receive label `-1`.
- Both image and text embeddings are L2-normalized before similarity computation.
- Image-text logits are computed as scaled cosine similarities plus the learned SigLIP logit bias.
- The matching objective is the SigLIP pairwise sigmoid loss, averaged across captions and images.
- The concept objective is weighted binary cross-entropy with logits over the 14,094-dimensional multi-label concept vector.
- The optimized loss is `matching loss + λ_concept × concept loss`.
- Optimization uses AdamW over all trainable parameters: the final 17 vision layers, concept bottleneck, logit scale, and logit bias.
- The learning rate is `1 × 10⁻⁴`, AdamW epsilon is `1 × 10⁻⁶`, and PyTorch's default AdamW weight decay of `0.01` is used.
- Gradients are cleared before each backward pass and clipped to a maximum global norm of 1.0 before each optimizer step.
- No learning-rate scheduler or warm-up procedure is used.
- Training is configured for a maximum of 15 epochs per concept-loss weight.
- Shape, finite-value, loss, and gradient sanity checks run during training. These checks ensure that frozen modules receive no gradients and all trainable gradients remain finite.
- Loss and tensor-shape diagnostics are printed for the first five batches of each training epoch.

## Loss-function calculation

- The training objective contains two components: an image-text matching loss and a multi-label concept-classification loss.
- Let $v_i$ denote the mean-pooled vision embedding for image $i$, and let $t_j$ denote the pooled embedding of caption $j$. Both embeddings are L2-normalized before their similarity is calculated:

$$
\hat{v}_i = \frac{v_i}{\lVert v_i \rVert_2},
\qquad
\hat{t}_j = \frac{t_j}{\lVert t_j \rVert_2}.
$$

- Every image is compared with every caption in the batch. The target matrix $y_{ij}$ is defined as:

$$
y_{ij} =
\begin{cases}
+1, & \text{if caption } j \text{ belongs to image } i, \\
-1, & \text{otherwise.}
\end{cases}
$$

- The image-caption matching logit is calculated as:

$$
z_{ij}
=
\exp\!\left(\min(s, \log 100)\right)
\left(\hat{v}_i^{\top}\hat{t}_j\right) + b,
$$

  where $s$ is the trainable SigLIP logit-scale parameter and $b$ is the trainable logit bias. Clamping the scale limits its exponentiated value to at most 100.

- The matching loss applies the SigLIP sigmoid objective to every image-caption pair:

$$
\mathcal{L}_{\mathrm{matching}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\left[
\frac{1}{C}
\sum_{j=1}^{C}
\log \sigma\!\left(y_{ij}z_{ij}\right)
\right],
$$

  where $B$ is the number of images in the batch, $C$ is the total number of captions in that batch, and $\sigma$ is the sigmoid function. Thus, matching pairs are encouraged to have positive logits and non-matching pairs are encouraged to have negative logits.

- The concept head predicts one logit $a_{ik}$ for each of the 14,094 concepts. Its target $c_{ik}$ is binary and indicates whether concept $k$ occurs in image $i$.
- To compensate for concept imbalance, each concept receives a positive-class weight calculated from the training split:

$$
w_k
=
\min\!\left(
\frac{N_{\mathrm{negative},k}}{N_{\mathrm{positive},k}},
20
\right).
$$

  If a concept does not have both positive and negative examples, its positive weight is set to 1.

- The concept loss is weighted binary cross-entropy with logits, averaged over the batch and all concepts:

$$
\mathcal{L}_{\mathrm{concept}}
=
\operatorname{mean}_{i,k}
\left[
-w_k c_{ik}\log\sigma(a_{ik})
-(1-c_{ik})\log\!\left(1-\sigma(a_{ik})\right)
\right].
$$

- The final loss used for backpropagation is:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{matching}}
+
\lambda_{\mathrm{concept}}\mathcal{L}_{\mathrm{concept}}.
$$

- The searched values of $\lambda_{\mathrm{concept}}$ are 0.0, 0.05, 0.1, 0.3, 0.5, and 1.0. When $\lambda_{\mathrm{concept}}=0$, the concept head does not affect the optimization objective even though its loss is still computed.
- If the optional predicate or object heads were enabled, their unweighted binary cross-entropy losses would be added as $\lambda_{\mathrm{predicate}}\mathcal{L}_{\mathrm{predicate}}$ and $\lambda_{\mathrm{object}}\mathcal{L}_{\mathrm{object}}$. Both heads are disabled in the current experiment.
- Training performs backpropagation through $\mathcal{L}_{\mathrm{total}}$. Validation calculates the same matching, concept, and total losses without gradients.
- Training and validation curves plot $\mathcal{L}_{\mathrm{total}}$; however, early stopping, best-epoch selection, and best-lambda selection use only $\mathcal{L}_{\mathrm{matching}}$ on the validation set.
- Final test evaluation disables all bottleneck losses, so its reported total loss is equal to $\mathcal{L}_{\mathrm{matching}}$.

## Hyperparameter search and checkpointing

- The concept-loss weight is searched over `λ_concept ∈ {0.0, 0.05, 0.1, 0.3, 0.5, 1.0}`.
- Every lambda run starts from the same initial vision-tower weights, concept-head weights, logit scale, and logit bias, enabling a fair comparison between lambda values.
- A new AdamW optimizer is initialized for every lambda value.
- After every training epoch, the model is evaluated on both the training and validation splits.
- Early stopping monitors validation matching loss, not the combined matching-plus-concept loss.
- Early stopping uses a patience of 5 epochs and requires an improvement greater than `1 × 10⁻³` to reset patience.
- Within each lambda run, the checkpoint with the lowest validation matching loss is retained, even when an improvement is smaller than the early-stopping threshold.
- The best lambda value is also selected using the lowest validation matching loss across all runs.
- A separate checkpoint is saved for every lambda value, for example `best_finetuned_siglip_concept_0.1.pt`.
- The overall best checkpoint is saved as `best_finetuned_siglip.pt`.
- Each checkpoint contains the vision-tower state, concept-bottleneck state, learned logit scale and bias, lambda value, best validation matching loss, best epoch, model identifier, and LoRA/QLoRA status.
- Training and validation total-loss curves are saved to `finetuning_lambda_curves.png` for all lambda values.

## Validation process

- Validation runs in evaluation mode and under `torch.no_grad()`, so model parameters are not updated.
- Validation uses the same image preprocessing, text encoding, mean pooling, matching-label construction, and loss functions as training.
- Both the matching loss and weighted concept loss are computed during validation.
- Reported validation total loss is the image-count-weighted average of the combined loss over the validation split.
- Although total validation loss is recorded and plotted, checkpoint selection, early stopping, and lambda selection are based exclusively on validation matching loss.

## Final test evaluation

- After hyperparameter selection, the overall best checkpoint is reloaded before test evaluation.
- Test evaluation is performed in evaluation mode without gradient computation.
- The concept bottleneck is disabled during final testing (`use_bottlenecks=False`). Consequently, the final test result measures only image-text matching loss and does not report concept classification performance.
- The final test output contains the average matching loss and total loss; because bottleneck losses are disabled, these two values are identical.

## Hyperparameter summary

- Base model: LLaVA-OneVision Qwen2 0.5B SI.
- Frozen text encoder: SigLIP SO400M Patch14-384.
- Input resolution: `384 × 384`.
- Batch size: `8`.
- Maximum epochs per lambda: `15`.
- Trainable vision layers: `17` of `26` (final 17 layers).
- Frozen vision layers: first `9`, plus the embedding layer.
- Number of concepts: `14,094`.
- Number of objects: `3,679` (head disabled).
- Number of predicates: `18` (head disabled).
- Optimizer: AdamW.
- Learning rate: `1 × 10⁻⁴`.
- AdamW epsilon: `1 × 10⁻⁶`.
- Weight decay: `0.01` (PyTorch default).
- Gradient clipping maximum norm: `1.0`.
- Concept-loss weights: `0.0`, `0.05`, `0.1`, `0.3`, `0.5`, and `1.0`.
- Early-stopping patience: `5` epochs.
- Early-stopping minimum improvement: `1 × 10⁻³`.
- Maximum concept positive-class weight: `20.0`.
- LoRA: disabled.
- QLoRA: disabled.
- Learning-rate scheduler: none.
- Random seed: not explicitly set in the current fine-tuning code.
